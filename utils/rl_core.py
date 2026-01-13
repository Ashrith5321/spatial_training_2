import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from collections import deque
from typing import Iterator
import ray
def collate_trajectories(trajectory_list: list[dict], device='cpu'):
    """
    Collates a list of trajectory dictionaries (numpy arrays) into a batched Tensor dictionary.
    Handles variable-length sequences via padding.

    Args:
        trajectory_list: List of dicts, where each dict has keys like 'actions', 'rewards', 'values'.
                         Shapes are expected to be (Seq_Len, ...) without batch dim.
        device: Target device for the tensors (default 'cpu', move to GPU before GAE).

    Returns:
        batch (dict): Dictionary of stacked, padded tensors (Batch, Max_Seq_Len, ...).
        response_mask (torch.Tensor): Boolean/Float mask (Batch, Max_Seq_Len), 1.0 for valid, 0.0 for pad.
    """
    if not trajectory_list:
        return {}, None

    # 1. Identify Keys and Dtypes
    # We infer expected types based on common RL keys to ensure PyTorch compatibility
    keys = trajectory_list[0].keys()
    batch = {}
    
    # Pre-calculate lengths for mask creation
    # Assumes 'actions' is always present and represents the timeline length
    lengths = [len(t['actions']) for t in trajectory_list]
    max_len = max(lengths)
    batch_size = len(trajectory_list)

    # 2. Iterate keys and Pad
    for key in keys:
        # Extract list of numpy arrays for this key
        arrays = [t[key] for t in trajectory_list]
        
        # Convert to Tensor (Automatically handles float/int inference)
        # Note: We force float32 for typical float types to avoid double precision overhead
        tensors = []
        for arr in arrays:
            t = torch.tensor(arr, device=device)
            if key in ['rewards', 'values', 'old_logprobs', 'logprobs']:
                t = t.float() # Ensure float32
            elif key in ['actions']:
                t = t.int()  # Ensure int32 for pointer
            tensors.append(t)
            
        # Pad Sequence
        # batch_first=True -> (Batch, Seq, ...)
        # padding_value=0 is standard (masked out anyway)
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)
        
        # Squeeze singleton dimensions if they exist (e.g. values being B,S,1)
        if padded.dim() == 3 and padded.shape[-1] == 1:
            padded = padded.squeeze(-1)
            
        batch[key] = padded

    batch['old_log_prob'] = batch['old_logprobs'].gather(2, batch['actions'].unsqueeze(-1)).squeeze(-1)
    # 3. Create Response Mask
    # 1 for valid tokens, 0 for padding
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.int, device=device)
    for i, length in enumerate(lengths):
        response_mask[i, :length] = 1
    batch['response_mask'] = response_mask
    return TensorDict(batch,batch_size=batch['old_log_prob'].shape[:2])

def collect_rollouts(
    sim_handles: list,
    vlm_handles: list,
    shard_iterator: Iterator[list[str]],
):
    """
    Collects a batch of trajectories. 
    The batch size is determined implicitly by the total number of episodes 
    contained in the shards yielded by `shard_iterator`.
    """
    
    # --- 1. State Initialization ---
    idle_vlms = deque(vlm_handles)
    ready_sims = deque()     # (sim_handle, reset_ref)
    
    # Future Tracking
    pending_resets = {}      # {reset_ref: sim_handle}
    active_episodes = {}     # {run_ref: "running"}
    pending_postproc = {}    # {pp_ref: vlm_handle}
    
    trajectory_buffer = []   # Results
    
    # State flags
    iterator_exhausted = False

    # --- 2. Bootstrap Sims ---
    # Assign initial work and trigger first reset
    print("Bootstrapping simulation workers...")
    for sim in sim_handles:
        try:
            shard = next(shard_iterator)
            sim.assign_shard.remote(shard)
            reset_ref = sim.reset.remote()
            pending_resets[reset_ref] = sim
        except StopIteration:
            iterator_exhausted = True
            # If we run out of shards during bootstrap, this worker is useless
            pass

    # --- 3. The Collection Loop ---
    # Run as long as there is active work or pending input
    while (active_episodes or pending_resets or pending_postproc or 
          (ready_sims and idle_vlms and not iterator_exhausted)):

        # A. Dispatcher
        # Try to pair idle VLMs with ready Sims
        while idle_vlms and ready_sims:
            vlm = idle_vlms.popleft()
            sim, init_ref = ready_sims.popleft()
            
            # Launch Episode
            run_ref = vlm.run_episode.remote(sim, init_ref)
            active_episodes[run_ref] = "running"
        
        # B. Wait for Events
        # We assume one of these maps has keys, otherwise the outer loop would have broken
        watch_list = list(pending_resets.keys()) + \
                     list(active_episodes.keys()) + \
                     list(pending_postproc.keys())
        
        # If nothing to watch, it means we are just waiting for the Dispatcher to find work
        # (unlikely in this logic flow, but good for safety)
        if not watch_list:
            break

        done_refs, _ = ray.wait(watch_list, num_returns=1)
        
        for ref in done_refs:
            
            # --- CASE 1: Sim Reset Complete ---
            if ref in pending_resets:
                sim = pending_resets.pop(ref)
                ready_sims.append((sim, ref)) # ref is the new initial_state
            
            # --- CASE 2: Episode Execution Complete ---
            elif ref in active_episodes:
                del active_episodes[ref]
                # Unpack results: VLM and Sim split paths here
                vlm_handle, sim_handle, is_exhausted, state_dict = ray.get(ref)
                
                # Path A: Simulator Lifecycle
                # Trigger reset immediately so it can start loading the next scene/episode
                try:
                    if is_exhausted:
                        # Fetch new work if needed
                        new_shard = next(shard_iterator)
                        sim_handle.assign_shard.remote(new_shard)
                    
                    # Always reset (either new shard or next ep in current shard)
                    new_reset_ref = sim_handle.reset.remote()
                    pending_resets[new_reset_ref] = sim_handle
                    
                except StopIteration:
                    iterator_exhausted = True
                    # This sim is now retired for this batch
                    pass

                # Path B: VLM Lifecycle
                # Send VLM to do the heavy lifting (tokenization/tensor packing)
                pp_ref = vlm_handle.postprocess_episode.remote()
                pending_postproc[pp_ref] = vlm_handle

            # --- CASE 3: Post-Processing Complete ---
            elif ref in pending_postproc:
                vlm_handle = pending_postproc.pop(ref)
                
                # Store Result
                traj_tuple = ray.get(ref) # (trajectory, tensors, metadata)
                trajectory_buffer.append(traj_tuple)
                
                # Recycle VLM
                idle_vlms.append(vlm_handle)

    return trajectory_buffer