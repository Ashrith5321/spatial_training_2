from longnav.config_schema import *
from longnav.utils.factories import ExpBootstrapper,get_shard_iterator
from longnav.env.env_base import DummyEnvActor
from longnav.utils.rollout_core import collect_rollouts
from longnav.utils.rl_core import collate_trajectories
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

import ray
import numpy as np
import json
cfg = RLConfig()
cfg.resources.osm_gb=28
cfg.resources.num_vlms=1
cfg.resources.vlm_gpu_fraction=0.3
cfg.resources.num_sims=2
cfg.resources.vlm_conda_env=None

cfg.vlm.attn_impl = "sdpa"
cfg.vlm.save_outputs=True
cfg.rollout.convo_start_template=[
        {"role": "user", "content": [{"type": "text", "text": "example substitution: $instr_or_goal"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ]
cfg.rollout.max_steps=16
cfg.training.rl_config.n_rollout=4
advantage_estimator_fn = get_adv_estimator_fn("reinforce_plus_plus")

cfg.task.run_name = "rl_step"
cfg.task.wandb_project = "longnav_smoke_test"
bootstrapper = ExpBootstrapper(cfg)

bootstrapper.setup_cluster()

trainers = bootstrapper.bootstrap_vlms_rl(training=True) 
sims = [ray.remote(DummyEnvActor).remote() for _ in range(2)]
try:
    wandb_actor,_ = bootstrapper.bootstrap_logger()
except Exception as e:
    wandb_actor = None
    print(f"Logger setup failed with error: {e}. Continuing without logger.")


trajectory_list = []

for i in range(3):
    rollout_list,result_list,log_list = collect_rollouts(sims,trainers,get_shard_iterator(0),bootstrapper.typed_cfg.training.rl_config.n_rollout) #
    trajectory_list += [tup[0] for tup in rollout_list]
    trajectory_list = trajectory_list[-bootstrapper.typed_cfg.training.rl_config.n_adv:]
    traj_batch = collate_trajectories(trajectory_list)
    model_inputs = [(tup[1],tup[2]) for tup in rollout_list]

    values = traj_batch.get("values",None)

    print("Computing Advantages")
    # config = bootstrapper.resolved_dict['training']['rl_config']
    # Note: compute_gae expects (B, T) inputs and returns (B, T)
    adv_tuple = advantage_estimator_fn(
        token_level_rewards=traj_batch['rewards'],
        values=values,
        response_mask=traj_batch['response_mask'],
        config = cfg.training.rl_config,
        # gamma=config.get('gamma', 0.99), # Fallback defaults if not in config
        # lam=config.get('lam', 0.95)
    )
    advantages, returns = adv_tuple[0],adv_tuple[1]
    traj_batch['advantages'] = advantages
    traj_batch['returns'] = returns

    print(f"traj batch shape: {traj_batch.shape}")
    traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout:] # only train on most recent.

    print("training step:")

    future_metadata = {}
    training_futures = []
    perm_indices = np.random.permutation(bootstrapper.typed_cfg.training.rl_config.n_rollout) # shuffle
    for batch_start in range(0, bootstrapper.typed_cfg.training.rl_config.n_rollout, bootstrapper.typed_cfg.resources.num_vlms):
        # Create futures for this specific "global step"
        # We map workers 0..N to data indices batch_start..batch_start+N
        step_futures = []
        for worker_idx, trainer in enumerate(trainers):
            global_idx = batch_start + worker_idx
            global_idx = perm_indices[global_idx]
            # Access the specific inputs and the sliced TensorDict for this index
            # We use global_idx to ensure we pull the correct corresponding data
            ref = trainer.train_rl_step.remote(
                    *model_inputs[global_idx], 
                    traj_batch[global_idx : global_idx + 1, traj_batch['response_mask'][global_idx].bool()]
                )
            step_futures.append(ref)
            future_metadata[ref] = global_idx
        training_futures.extend(step_futures)

    print(f"Dispatched {len(training_futures)} training tasks for epoch")
    # ------------------------------------- monitor training live ---------------------------------
    pending_futures = training_futures
    total_tasks = len(pending_futures)
    completed_count = 0

    while pending_futures:
        # 1. Block until at least one future is ready
        ready_refs, pending_futures = ray.wait(pending_futures, num_returns=1)
        
        # 2. Process the ready future(s)
        for ref in ready_refs:
            # We catch exceptions here to prevent one failed batch from crashing the loop
            try:
                result = ray.get(ref)
                rollout_idx = future_metadata[ref]

                batch_row = traj_batch[rollout_idx] 
                valid_mask = batch_row['response_mask'].bool()
                traj_stats = batch_row[valid_mask] 


                # 4. Log
                rollout_stats = {
                    "rollout/ep_rew": traj_stats['rewards'].sum().item(),
                    "rollout/ep_len": valid_mask.sum().item(),
                    # "rollout/success": traj_stats['success'].max().item(), 
                    # "rollout/spl": traj_stats['spl'].max().item(),
                    "rollout/ep_rtn": traj_stats['returns'].mean().item(),
                    "rollout/rtn_var": traj_stats['returns'].var().item(),
                    # "rollout/global_cycle": global_cycle
                }
                try:
                    critic_mse = ((traj_stats['baseline']-traj_stats['returns'])**2).mean()
                    # naive_mse = ((traj_stats['returns'] - global_return_mean)**2).mean().item()
                    rollout_stats |= {
                        "rollout/baseline_mse":critic_mse,
                        # "rollout/naive_mse":naive_mse
                    }
                except:
                    print("cannot compute baseline metric")
                result |= rollout_stats
                log_ref = log_list[rollout_idx]
                try:
                    # 1. Try to get the path with a short timeout (e.g., 0.1s)
                    # If the Sim Worker is done, this is instant.
                    log_path = ray.get(log_ref, timeout=30.0)
                    
                    with open(log_path, 'r') as f:
                        vlm_log_dict = json.load(f)
                    result |= vlm_log_dict
                    
                except ray.exceptions.GetTimeoutError:
                    # 2. If Sim Worker is stuck, log a warning but DO NOT FREEZE training
                    print(f"Log file for rollout {rollout_idx} not ready (Sim Worker I/O Lag). Skipping detailed logs.")
                except Exception as e:
                    print(f"Failed to read log file: {e}")
                if wandb_actor is not None:
                    wandb_actor.log_row.remote(result)
                completed_count += 1
                
                print(f"[{completed_count}/{total_tasks}] Complete. {result}")
            except Exception as e:
                print(f"Error in training future: {e}")
                completed_count += 1
                print(f"[{completed_count}/{total_tasks}] Complete with error. Check logs for details.")
ray.get(wandb_actor.close.remote())