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
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, length in enumerate(lengths):
        response_mask[i, :length] = 1
    batch['response_mask'] = response_mask
    return TensorDict(batch,batch_size=batch['old_log_prob'].shape[:2])

def collect_rollouts(
    sim_handles: list,
    vlm_handles: list,
    shard_iterator: Iterator[list[str]],
    target_episodes: int = float('inf')
) -> list:
    """
    Orchestrates the RL collection pipeline.
    Structurally identical to run_inference_driver, except VLM recycling 
    is delayed until post-processing completes.
    """

    # --- 1. Initialize Pools ---
    idle_vlms = deque(vlm_handles)
    ready_sims = deque() 

    # --- 2. Tracking Futures ---
    pending_resets = {}   # reset_ref -> sim_handle
    active_episodes = {}  # sup_ref -> "running"
    
    # [DIFFERENCE]: New state for VLM post-processing
    pending_postproc = {} # pp_ref -> vlm_handle 

    trajectory_buffer = []
    iterator_exhausted = False

    # --- 3. Bootstrap: Initial Sharding & Resets (IDENTICAL) ---
    print(f"Bootstrapping: Initializing {len(sim_handles)} environments...")
    for sim_handle in sim_handles:
        try:
            initial_shard = next(shard_iterator)
            sim_handle.assign_shard.remote(initial_shard)
            reset_ref = sim_handle.reset.remote()
            pending_resets[reset_ref] = sim_handle   
        except StopIteration:
            iterator_exhausted = True
            print("Warning: Not enough shards for all workers during bootstrap.")
            pass

    
    # Helper to check if we should keep the loop alive
    def has_work():
        # 1. Are tasks currently running?
        is_active = len(active_episodes) > 0 or len(pending_resets) > 0 or len(pending_postproc) > 0
        
        # 2. Can we launch new tasks? (Resources available AND Target not met)
        potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)
        can_launch = (len(idle_vlms) > 0 and len(ready_sims) > 0)
        should_launch = can_launch and (potential < target_episodes) and (not iterator_exhausted)
        
        return is_active or should_launch

    # --- Event Loop ---
    while has_work():
        
        # A. Dispatch (IDENTICAL)
        total_potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)
        
        while (idle_vlms and ready_sims and total_potential < target_episodes):
            vlm = idle_vlms.popleft()
            sim, init_state_ref = ready_sims.popleft()
            
            sup_ref = vlm.run_episode.remote(sim, init_state_ref)
            active_episodes[sup_ref] = "running"
            total_potential +=1


        # B. Wait for Events
        all_watch_refs = list(pending_resets.keys()) + \
                         list(active_episodes.keys()) + \
                         list(pending_postproc.keys()) # Added check
        
        if not all_watch_refs:
            break

        ready_refs, _ = ray.wait(all_watch_refs, num_returns=1)
        
        for ref in ready_refs:
            
            # --- CASE 1: Reset Finished (IDENTICAL) ---
            if ref in pending_resets:
                sim_handle = pending_resets.pop(ref)
                ready_sims.append((sim_handle, ref))
            
            # --- CASE 2: Episode Finished (MODIFIED) ---
            elif ref in active_episodes:
                del active_episodes[ref]
                
                # Unpack results
                vlm, hab, is_exhausted, state = ray.get(ref)
                
                # [DIFFERENCE]: VLM does NOT go to idle_vlms yet.
                # It goes to post-processing.
                pp_ref = vlm.postprocess_episode.remote()
                pending_postproc[pp_ref] = vlm

                # Sim Logic: [IDENTICAL to Inference]
                try:
                    if is_exhausted:
                        new_shard = next(shard_iterator)
                        hab.assign_shard.remote(new_shard)
                    
                    new_reset_ref = hab.reset.remote()
                    pending_resets[new_reset_ref] = hab
                except StopIteration:
                    # No more work. Retire the Habitat worker.
                    iterator_exhausted = True
                    ray.get(hab._flush_logs_to_disk.remote()) 
                    pass
            
            # --- CASE 3: Post-Processing Finished (NEW) ---
            elif ref in pending_postproc:
                vlm = pending_postproc.pop(ref)
                
                # Get the packed trajectory data
                traj_tuple = ray.get(ref)
                trajectory_buffer.append(traj_tuple)
                
                # [DIFFERENCE]: NOW the VLM is recycled
                idle_vlms.append(vlm)
                print(f"Collected episode {len(trajectory_buffer)}")
    return trajectory_buffer