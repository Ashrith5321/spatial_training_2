import torch

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from collections import deque
from typing import Iterator,Optional
import ray
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est

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
            if key in ['rewards', 'values', 'old_logprobs', 'logprobs', 'ref_logprobs']:
                t = t.float() # Ensure float32
            elif key in ['actions']:
                t = t.long()  # Ensure int32 for pointer
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
    target_episodes: int = float('inf'),
    postprocess_kwargs = {"return_inputs":True, "eval":False}
) -> tuple[list,list,list]:
    """
    Orchestrates the RL collection pipeline.

    returns: trajectory buffer, result list, log list, indexed by dispatch_id
    """

    # --- 1. Initialize Pools ---
    idle_vlms = deque(vlm_handles)
    ready_sims = deque() 

    # --- 2. Tracking Futures ---
    pending_resets = {}   # reset_ref -> sim_handle
    active_episodes = {}  # ep_ref -> dispatch_id
    
    # VLM post-processing
    pending_postproc = {} # pp_ref -> vlm_handle, dispatch_id 
    # Sim logging
    pending_logs = {} # log_ref -> sim_handle, dispatch_id

    trajectory_buffer = []
    trajectory_ids = []

    result_dict = {}
    log_dict = {}
    iterator_exhausted = False

    # --- 3. Bootstrap: Initial Sharding & Resets (IDENTICAL) ---
    for sim_handle in sim_handles:
        try:
            if ray.get(sim_handle.is_exhausted.remote()):
                initial_shard = next(shard_iterator)
                sim_handle.assign_shard.remote(initial_shard)
            reset_ref = sim_handle.reset.remote()
            pending_resets[reset_ref] = sim_handle   
        except StopIteration:
            iterator_exhausted = True
            print("Warning: Not enough shards for all workers during bootstrap.")
            pass
    print(f"Bootstrapping: Initializing {len(sim_handles)} environments...")

    
    # Helper to check if we should keep the loop alive
    def has_work():
        # 1. Are tasks currently running?
        is_active = len(active_episodes) > 0 or len(pending_postproc) > 0
        # 2. Do we still want to launch new tasks (now or in the future)? (Resources available AND Target not met)
        potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)
        want_launch = (potential < target_episodes) and (not iterator_exhausted) 
        return is_active or want_launch
    
    dispatch_counter = 0
    # --- Event Loop ---
    while has_work():
        
        # A. Dispatch (IDENTICAL)
        total_potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)
        
        while (idle_vlms and ready_sims and total_potential < target_episodes):
            vlm = idle_vlms.popleft()
            sim, init_state_ref = ready_sims.popleft()
            ep_ref = vlm.run_episode.remote(sim, init_state_ref)
            active_episodes[ep_ref] = dispatch_counter
            dispatch_counter +=1
            total_potential +=1


        # B. Wait for Events
        all_watch_refs = list(pending_resets.keys()) + \
                         list(active_episodes.keys()) + \
                         list(pending_postproc.keys()) + \
                         list(pending_logs.keys())
        
        if not all_watch_refs:
            break

        ready_refs, _ = ray.wait(all_watch_refs, num_returns=1)
        
        for ref in ready_refs:
            
            # --- CASE 1: Reset Finished ---
            if ref in pending_resets:
                sim_handle = pending_resets.pop(ref)
                ready_sims.append((sim_handle, ref))
            
            # --- CASE 2: Episode Finished ---
            elif ref in active_episodes:
                dispatch_id =  active_episodes.pop(ref)
                # Unpack results
                vlm, sim, is_exhausted, state = ray.get(ref)
                result_dict[dispatch_id] = state

                # send vlm and sim to post episode processing
                pp_ref = vlm.postprocess_episode.remote(**postprocess_kwargs)
                pending_postproc[pp_ref] = vlm,dispatch_id

                log_ref = sim._flush_logs_to_disk.remote()
                pending_logs[log_ref] = sim,dispatch_id,is_exhausted                
            
            # --- CASE 3: VLM Post-Processing Finished ---
            elif ref in pending_postproc:
                vlm,dispatch_id = pending_postproc.pop(ref)    
                trajectory_buffer.append(ref)
                trajectory_ids.append(dispatch_id)
                idle_vlms.append(vlm)
                print(f"Collected episode {len(trajectory_buffer)}")

            # --- CASE 4: Sim Log Flush Finished
            elif ref in pending_logs:
                sim,dispatch_id,is_exhausted = pending_logs.pop(ref)
                log_dict[dispatch_id] = ref # save the path to the log
                # send the sim to reset/reshard so it can start working again asap
                try:
                    if is_exhausted:
                        new_shard = next(shard_iterator)
                        sim.assign_shard.remote(new_shard)
                    new_reset_ref = sim.reset.remote()
                    pending_resets[new_reset_ref] = sim
                except StopIteration:
                    # No more work. Retire the Habitat worker.
                    iterator_exhausted = True
                    pass
    rollouts = [t for _, t in sorted(zip(trajectory_ids, trajectory_buffer))]
    log_dict |={v[1]:k for k,v in pending_logs.items()}
    num_rollouts = len(rollouts)
    result_list = [result_dict[i] for i in range(num_rollouts)]
    log_list = [log_dict[i] for i in range(num_rollouts)]
    return ray.get(rollouts), result_list, log_list


def apply_hybrid_splitting(batch, dagger_percentile=0.6, only_failures=True, stop_action_id=0,max_dagger_steps=100):
    """
    Mutates batch in-place to separate PPO and DAgger samples.
    
    Selection Logic:
    1. Rescue: Selects worst trajectories based on min(returns).
       - If only_failures=True, checks terminal reward.
       - Phases out DAgger automatically if failure count drops to 0.
    2. Surgical: Always catches False Positive/Negative stops everywhere.
    
    Args:
        batch (TensorDict): Mutated in-place. 'response_mask' (int/long) is updated. 'dagger_mask' (int/long) is added.
    """
    device = batch['actions'].device
    batch_size, seq_len = batch['actions'].shape
    
    # Use boolean view for logic, but keep original dtype for storage
    # response_mask is (B, T), 1=valid, 0=pad
    valid_mask_bool = batch['response_mask'].bool()
    
    # --- 1. Identify Failures (Terminal Reward Check) ---
    lengths = valid_mask_bool.sum(dim=1).long()
    last_indices = (lengths - 1).clamp(min=0)
    
    # Check reward at the final step
    # Assumption: Success reward > 0.1 (accounting for float slack)
    terminal_rewards = batch['rewards'][torch.arange(batch_size, device=device), last_indices]
    is_failure = terminal_rewards <= 0.1 
    
    # --- 2. Score Trajectories (Min Step-Level Return) ---
    masked_returns = batch['returns'].clone()
    masked_returns[~valid_mask_bool] = float('inf')
    
    # (B,) Scores: Lower is worse
    trajectory_scores = masked_returns.min(dim=1).values
    
    # --- 3. Trajectory Selection (The Rescue) ---
    traj_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    if only_failures:
        # Filter: Only consider failed indices
        failed_indices = torch.nonzero(is_failure).squeeze(-1)
        n_failures = len(failed_indices)
        
        if n_failures > 0:
            # Calculate k based on the FAILURE count
            k = int(n_failures * dagger_percentile)
            
            if k > 0:
                # Get scores for failed episodes only
                failed_scores = trajectory_scores[failed_indices]
                # Find worst k among failures
                _, top_k_sub_indices = torch.topk(failed_scores, k, largest=False)
                # Map back to global batch indices
                target_indices = failed_indices[top_k_sub_indices]
                traj_mask[target_indices] = True
    else:
        # Standard: Bottom N% of ALL episodes
        k = int(batch_size * dagger_percentile)
        if k > 0:
            _, target_indices = torch.topk(trajectory_scores, k, largest=False)
            traj_mask[target_indices] = True
    # --- 4. Mask Generation (Integrated Clipping) ---
    # Create a time index tensor (1, T) to compare against max_dagger_steps
    time_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Generate the base DAgger mask:
    # 1. Episode was selected (traj_mask)
    # 2. Step is real data (valid_mask_bool)
    # 3. Step is within the "useful" supervision window (time_indices < max_dagger_steps)
    dagger_mask_bool = (
        traj_mask.unsqueeze(1) & 
        valid_mask_bool & 
        (time_indices < max_dagger_steps)
    )


    # --- 4. Surgical Selection (FP/FN Stops) ---
    actions = batch['actions']
    oracle_actions = batch['oracle_actions']
    
    if oracle_actions.is_floating_point():
        oracle_indices = oracle_actions.argmax(dim=-1)
    else:
        oracle_indices = oracle_actions

    # FP: Agent=STOP (0), Oracle!=STOP
    fp_mask = (actions == stop_action_id) & (oracle_indices != stop_action_id)
    # FN: Agent!=STOP, Oracle=STOP (0)
    fn_mask = (actions != stop_action_id) & (oracle_indices == stop_action_id)
    
    surgical_mask = (fp_mask | fn_mask) & valid_mask_bool
    
    # --- 5. Final Mutating Merge ---
    # Merge masks
    dagger_mask_bool = dagger_mask_bool | surgical_mask
    
    # Update PPO Mask (Remove DAgger steps from PPO)
    # Ensure we maintain the original INT dtype
    ppo_mask_bool = valid_mask_bool & (~dagger_mask_bool)
    
    batch['response_mask'] = ppo_mask_bool.to(batch['response_mask'].dtype)
    batch['dagger_mask'] = dagger_mask_bool.to(batch['response_mask'].dtype)
    
    return batch

@register_adv_est("reinforce_plus_plus_linear_time_aware")
def compute_reinforce_plus_plus_linear_time_aware_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    config: Optional[AlgoConfig] = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute REINFORCE++ advantages using a Linear Time-Decay Baseline.
    Fixes applied:
    1. Numerical Stability: Regressions calculated in float32 to prevent cancellation.
    2. End-Alignment: Uses negative time (t - seq_len) to align goal states.
    """
    assert config is not None
    gamma = config.gamma
    device = token_level_rewards.device
    
    # 1. Compute Standard Discounted Returns (G_t)
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]

        # 2. Prepare Data for Linear Regression
        bs, seq_len = returns.shape
        seq_lengths = response_mask.sum(dim=-1)
        
        # Create time indices. IMPORTANT: Use float32 for the grid to avoid casting later
        time_indices = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0).expand(bs, seq_len)
        
        # ALIGNMENT FIX: Shift time so the last step is roughly 0 (or -1). 
        # t_new goes from [-Length, 0]. 
        # This aligns the "End of Episode" across the batch.
        time_indices = time_indices - seq_lengths.unsqueeze(1)

        # Flatten and Mask
        valid_mask = response_mask.bool()
        
        # NUMERICAL STABILITY FIX: Ensure inputs to regression are float32
        x_flat = time_indices[valid_mask].to(torch.float32) 
        y_flat = returns[valid_mask].to(torch.float32)
        
        N = x_flat.numel()
        
        if N > 1:
            # 3. Closed-Form Linear Least Squares (Weighted by data points)
            sum_x = x_flat.sum()
            sum_y = y_flat.sum()
            sum_xy = (x_flat * y_flat).sum()
            sum_xx = (x_flat * x_flat).sum()

            # Denominator: N*Var(x). 
            # Adding epsilon is crucial for short sequences where Var(x) might be 0.
            denominator = N * sum_xx - sum_x * sum_x + 1e-6
            
            m = (N * sum_xy - sum_x * sum_y) / denominator
            c = (sum_y - m * sum_x) / N

            # 4. Compute Baseline 
            # Cast back to original dtype for subtraction if needed, or keep as float32 for precision
            baseline = m * time_indices + c
            
            # The Advantage is the Residual (Actual - Predicted)
            # We cast baseline to the return's dtype (e.g., bfloat16) to match
            advantages = returns - baseline.to(returns.dtype)
        else:
            advantages = returns

        # 5. Whiten the Residuals (Standardize variance)
        advantages = verl_F.masked_whiten(advantages, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

@register_adv_est("reinforce_plus_plus_geometric_time_aware")
def compute_reinforce_plus_plus_geometric_time_aware_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    config: Optional[AlgoConfig] = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute REINFORCE++ advantages using a Linear Time-Decay Baseline.
    Fixes applied:
    1. Numerical Stability: Regressions calculated in float32 to prevent cancellation.
    2. End-Alignment: Uses negative time (t - seq_len) to align goal states.
    """
    assert config is not None
    gamma = config.gamma
    device = token_level_rewards.device
    
    # 1. Compute Standard Discounted Returns (G_t)
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]

        # 2. Prepare Data for Linear Regression
        bs, seq_len = returns.shape
        seq_lengths = response_mask.sum(dim=-1)

        # Create time indices. IMPORTANT: Use float32 for the grid to avoid casting later
        time_indices = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0).expand(bs, seq_len)
        # Calculate Discounted Horizon Basis: S(h) = (1 - gamma^h) / (1 - gamma)
        # Horizon (Time-to-Go) = Total Length - Current Time Step
        horizon = seq_lengths.unsqueeze(1) - time_indices
        # Transform input feature to match the geometric shape of discounted returns
        # This ensures the regression fits the "Curve" of returns, not just a Line.
        if abs(1.0 - gamma) < 1e-6:
            time_indices = horizon
        else:
            time_indices = (1.0 - torch.pow(gamma, horizon)) / (1.0 - gamma)

        # Flatten and Mask
        valid_mask = response_mask.bool()
        
        # NUMERICAL STABILITY FIX: Ensure inputs to regression are float32
        x_flat = time_indices[valid_mask].to(torch.float32) 
        y_flat = returns[valid_mask].to(torch.float32)
        
        N = x_flat.numel()
        
        if N > 1:
            # 3. Closed-Form Linear Least Squares (Weighted by data points)
            sum_x = x_flat.sum()
            sum_y = y_flat.sum()
            sum_xy = (x_flat * y_flat).sum()
            sum_xx = (x_flat * x_flat).sum()

            # Denominator: N*Var(x). 
            # Adding epsilon is crucial for short sequences where Var(x) might be 0.
            denominator = N * sum_xx - sum_x * sum_x + 1e-6
            
            m = (N * sum_xy - sum_x * sum_y) / denominator
            c = (sum_y - m * sum_x) / N

            # 4. Compute Baseline 
            # Cast back to original dtype for subtraction if needed, or keep as float32 for precision
            baseline = m * time_indices + c
            
            # The Advantage is the Residual (Actual - Predicted)
            # We cast baseline to the return's dtype (e.g., bfloat16) to match
            advantages = returns - baseline.to(returns.dtype)
        else:
            advantages = returns

        # 5. Whiten the Residuals (Standardize variance)
        advantages = verl_F.masked_whiten(advantages, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

def _generate_gaussian_kernel_1d(sigma: float, kernel_size: int, device: torch.device) -> torch.Tensor:
    """Generates a 1D Gaussian kernel for convolution."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).view(1, 1, -1)

@register_adv_est("reinforce_plus_plus_distance_kernel")
def compute_reinforce_plus_plus_distance_kernel_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    config: Optional[AlgoConfig] = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes REINFORCE++ advantages using a Distance-Based Kernel Baseline.
    
    This acts as a non-parametric critic: V(s) ~= E[Return | Distance_to_Goal].
    It solves "Consumption Bias" by comparing efficient agents (low dist, low return)
    only against other agents with similar remaining work.
    
    Requires 'distances' tensor in kwargs (shape: [Batch, Seq]).
    """
    assert config is not None
    # Check for required distance feature
    distances = kwargs.get('distances')
    if distances is None:
        # Fallback to info if packed differently, or raise error
        if 'info' in kwargs and 'distance_to_goal' in kwargs['info']:
             distances = kwargs['info']['distance_to_goal']
        else:
             raise ValueError("Advantage estimator 'distance_kernel' requires 'distances' or 'info['distance_to_goal']' in kwargs.")

    gamma = config.gamma
    device = token_level_rewards.device
    bs, seq_len = token_level_rewards.shape
    
    with torch.no_grad():
        # 1. Standard Discounted Return Calculation (G_t)
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(seq_len)):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]

        # ---------------------------------------------------------------------
        # 2. Distance-Based Kernel Regression (via Efficient Binning)
        # ---------------------------------------------------------------------
        
        # A. Configuration
        # Resolution: 0.1 means 10cm buckets. 
        # Sigma: 0.5 means the kernel spreads influence over +/- 1.5m roughly (3 sigma).
        bin_resolution = 0.1 
        kernel_sigma_meters = 0.5
        if config.distance_kernel_sigma is not None:
            kernel_sigma_meters = config.distance_kernel_sigma 
        dul,dulq = 0,0 #zero init
        if config.distance_clip_max is not None:
            dul = config.distance_clip_max
        if config.distance_clip_percentile is not None:
            dulq = torch.quantile(distances, config.distance_clip_percentile)
        dul = max(dul,dulq)
        if dul>1e-5:
            distances = torch.clamp(distances,max=dul)
            print(f"using distance upper limit {dul}")
        # Convert sigma from meters to bins
        sigma_bins = kernel_sigma_meters / bin_resolution
        kernel_size = int(8 * sigma_bins) + 1 # 8 sigma coverage
        
        # B. Discretize Distances
        # We handle the dynamic range of the batch automatically.
        # Apply mask: we don't want to bin padding (usually dist=0 or inf)
        valid_distances = distances * response_mask
        max_dist = valid_distances.max()
        
        # Create indices
        dist_indices = (valid_distances / bin_resolution).long()
        num_bins = int(max_dist / bin_resolution) + 1 + (kernel_size // 2) # Add padding room
        
        # C. Scatter to Bins (Aggregating Returns by Distance)
        # We need flattened views for scatter_add
        flat_indices = dist_indices.view(-1)
        flat_returns = (returns * response_mask).view(-1)
        flat_counts = response_mask.view(-1) # 1.0 for valid, 0.0 for pad
        
        # Accumulators
        bin_sum = torch.zeros(num_bins, device=device, dtype=torch.float32)
        bin_count = torch.zeros(num_bins, device=device, dtype=torch.float32)
        
        bin_sum.scatter_add_(0, flat_indices, flat_returns.float())
        bin_count.scatter_add_(0, flat_indices, flat_counts.float())
        
        # D. Kernel Smoothing (1D Convolution over Distance)
        # View as (1, 1, Length) for conv1d
        input_sum = bin_sum.view(1, 1, -1)
        input_count = bin_count.view(1, 1, -1)
        
        kernel = _generate_gaussian_kernel_1d(sigma_bins, kernel_size, device)
        pad = kernel_size // 2
        
        # Use replicate padding to handle boundaries (0m and Max Distance) gracefully
        padded_sum = F.pad(input_sum, (pad, pad), mode=config.distance_pad_mode, value=config.distance_pad_val) #'constant', "replicate"
        padded_count = F.pad(input_count, (pad, pad), mode=config.distance_pad_mode, value=config.distance_pad_val)
        
        smoothed_sum = F.conv1d(padded_sum, kernel)
        smoothed_count = F.conv1d(padded_count, kernel)
        
        # E. Compute Baseline Table
        # Baseline[bin] = Avg Return for that distance
        baseline_table = smoothed_sum / (smoothed_count + 1e-8) # (1, 1, num_bins)
        baseline_table = baseline_table.view(-1) # (num_bins,)
        
        # F. Project back to Token Space (Gather)
        # Map bin values back to original token positions
        baseline_flat = baseline_table[flat_indices]
        
        baseline = baseline_flat.view(bs, seq_len)
        
        # 3. Compute Advantage
        # A = G_t - b(dist_t)
        advantages = returns - baseline.to(returns.dtype)
        
        # ---------------------------------------------------------------------
        # 4. Global Normalization (REINFORCE++ Standard)
        # ---------------------------------------------------------------------
        advantages = verl_F.masked_whiten(advantages, response_mask)
        advantages = advantages * response_mask

    return advantages, returns, baseline_table

@register_adv_est("reinforce_plus_plus_distance_kernel_var_norm")
def compute_reinforce_plus_plus_distance_kernel_var_norm_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    config: Optional[AlgoConfig] = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Advantage using Distance-Conditional Mean AND Variance.
    
    1. Estimates Mean: mu(d) = E[G | d]
    2. Estimates Variance: sigma^2(d) = E[G^2 | d] - mu(d)^2
    3. Normalizes Locally: A = (G - mu(d)) / sigma(d)
    
    This solves the signal drowning problem where high-variance 'far' states 
    dominate the global normalization, suppressing precise 'near' state signals.
    """
    assert config is not None
    # Check for required distance feature
    distances = kwargs.get('distances')
    if distances is None:
         if 'info' in kwargs and 'distance_to_goal' in kwargs['info']:
             distances = kwargs['info']['distance_to_goal']
         else:
             raise ValueError("Requires 'distances' in kwargs.")

    gamma = config.gamma
    device = token_level_rewards.device
    bs, seq_len = token_level_rewards.shape
    
    with torch.no_grad():
        # 1. Standard Discounted Return Calculation
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        for t in reversed(range(seq_len)):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]

        # ---------------------------------------------------------------------
        # 2. Kernel Estimation of Mean AND Second Moment
        # ---------------------------------------------------------------------
        
        bin_resolution = 0.1 
        kernel_sigma_meters = 0.5 
        if config.distance_kernel_sigma is not None:
            kernel_sigma_meters = config.distance_kernel_sigma 
        dul,dulq = 0,0 #zero init
        if config.distance_clip_max is not None:
            dul = config.distance_clip_max
        if config.distance_clip_percentile is not None:
            dulq = torch.quantile(distances, config.distance_clip_percentile)
        dul = max(dul,dulq)
        if dul>1e-5:
            distances = torch.clamp(distances,max=dul)
            print(f"using distance upper limit {dul}")

        sigma_bins = kernel_sigma_meters / bin_resolution

        kernel_size = int(6 * sigma_bins) + 1 
        
        valid_distances = distances * response_mask
        max_dist = valid_distances.max()
        dist_indices = (valid_distances / bin_resolution).long()
        num_bins = int(max_dist / bin_resolution) + 1 + (kernel_size // 2)
        
        flat_indices = dist_indices.view(-1)
        flat_returns = (returns * response_mask).view(-1)
        flat_sq_returns = (flat_returns ** 2) # Pre-compute squares
        flat_counts = response_mask.view(-1)
        
        # Accumulators for E[G], E[G^2], and N
        bin_sum = torch.zeros(num_bins, device=device)
        bin_sq_sum = torch.zeros(num_bins, device=device)
        bin_count = torch.zeros(num_bins, device=device)
        
        bin_sum.scatter_add_(0, flat_indices, flat_returns.float())
        bin_sq_sum.scatter_add_(0, flat_indices, flat_sq_returns.float())
        bin_count.scatter_add_(0, flat_indices, flat_counts.float())
        
        # Kernel Smoothing (1D Conv)
        kernel = _generate_gaussian_kernel_1d(sigma_bins, kernel_size, device)
        pad = kernel_size // 2
        
        # Pad and Convolve all three statistics
        # Note: We group operations for efficiency
        stacked_inputs = torch.stack([bin_sum, bin_sq_sum, bin_count]).unsqueeze(0) # (1, 3, bins)
        padded_inputs = F.pad(stacked_inputs, (pad, pad), mode=config.distance_pad_mode, value=config.distance_pad_val)
        
        # We need a grouped convolution or just loop. Since it's only 3 channels, 
        # using groups=1 with repeated kernel is easy, or just simple loop.
        # Let's use simple loop for clarity as cost is negligible.
        smoothed_sum = F.conv1d(padded_inputs[:, 0:1], kernel)
        smoothed_sq_sum = F.conv1d(padded_inputs[:, 1:2], kernel)
        smoothed_count = F.conv1d(padded_inputs[:, 2:3], kernel)
        
        # Compute Conditional Stats
        # E[G|d]
        mean_table = smoothed_sum / (smoothed_count + 1e-8)
        
        # E[G^2|d]
        sq_mean_table = smoothed_sq_sum / (smoothed_count + 1e-8)
        
        # Var[G|d] = E[G^2] - (E[G])^2
        # Clamp to 0 to handle floating point noise causing negative variance
        var_table = torch.clamp(sq_mean_table - (mean_table ** 2), min=0.0)
        std_table = torch.sqrt(var_table)
        
        # Map back to tokens
        mean_table = mean_table.view(-1)
        std_table = std_table.view(-1)
        
        mu_d = mean_table[flat_indices].view(bs, seq_len)
        sigma_d = std_table[flat_indices].view(bs, seq_len)
        
        # 3. Compute Locally Normalized Advantage
        # A = (G - mu(d)) / (sigma(d) + epsilon)
        advantages = (returns - mu_d) / (sigma_d + 1e-5)
        # 4. Optional Global Polish
        # Since we effectively Z-scored everything locally, the global mean is roughly 0 
        # and global std is roughly 1. 
        # We perform one final lightweight whitening to catch any outliers or 
        # systematic drift, ensuring strict stability for PPO clipping.
        advantages = verl_F.masked_whiten(advantages, response_mask)
        advantages = advantages * response_mask

    return advantages, returns, mean_table, std_table