import os 
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'
import hydra
from hydra import compose, initialize
from conf.register_configs import register_configs
from utils.factories import ExpBootstrapper,get_shard_iterator,get_console_logger
from utils.inference_core import run_inference_driver, create_shard_iterator
import ray
from utils.tensor_utils import TensorPacker
from utils.rl_core import collate_trajectories,collect_rollouts
import torch
import numpy as np

# Placeholder import for verl GAE
try:
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return,get_adv_estimator_fn,AdvantageEstimator,POLICY_LOSS_REGISTRY
except ImportError:
    # If you haven't installed verl in this env yet, paste the function definition 
    # from the previous turn here.
    raise ImportError("Please install verl or paste the compute_gae_advantage_return function definition.")

# Register our custom variants and resolvers
register_configs()

# Initialize Hydra manually (replaces @hydra.main)
# 'config_path' is relative to this notebook
with initialize(version_base=None, config_path="conf"):
    # Here you can list overrides just like you would on the CLI
    cfg = compose(config_name="rl_config", overrides=[
        "task.run_name=anti_collision_kld",
        "task.wandb_project=rl_dev",
        "rollout.max_steps=200",
        "vlm.save_outputs=True", # need this for RL
        "task.subset_label=sample400",
        # "sim.split=train_mini",
        # "task.shard_size=0", # no sharding, fulldataset for everyone
        "resources.num_vlms=3",
        "resources.num_sims=4",
        "resources.master_port=25653",
        f"training.rl_config.advantage_estimator={AdvantageEstimator.REINFORCE_PLUS_PLUS}",
        "training.grad_accum_steps=4",
        "training.rl_config.gamma=0.6",
        "training.learning_rate=1e-5",
        "sim.fp_guard=false",
        "training.rl_config.use_value=false",

        # "training.rl_config.policy_loss_name="
    ])

advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)

print(f"Model ID: {cfg.vlm.model_id}")
bootstrapper = ExpBootstrapper(cfg)
logger = get_console_logger()

bootstrapper.setup_cluster()
trainers = bootstrapper.bootstrap_vlms_rl() #allocate vlms first to prevent out of room issues
wandb_actor = bootstrapper.bootstrap_logger()
sim_logger = None
sims = bootstrapper.bootstrap_sims(sim_logger)
shard_futures = [sim.assign_shard.remote(None) for sim in sims]
ray.get(shard_futures)

#
# # 3. Prepare Data Shards (using simple helper)
# shard_iter = get_shard_iterator(
#     subset_label= cfg.task.subset_label,
#     episode_json= cfg.task.episode_json,
#     shard_size=cfg.task.shard_size,
#     logger=logger
# )
try:
    for i in range(100):
        shard_iter = get_shard_iterator(
            subset_label= cfg.task.subset_label,
            episode_json= cfg.task.episode_json,
            shard_size=cfg.task.shard_size,
            logger=logger
        )
        # ------------------------------------------- rollouts ------------------------------------------
        logger.info("Starting rollout collection!")

        trajectory_list = collect_rollouts(sims,trainers,shard_iter,target_episodes=12) #
        print("done collecting")
        num_vlms = len(trainers)
        num_trajectories = len(trajectory_list)
        remainder = num_trajectories % num_vlms

        if remainder != 0:
            logger.warning(
                f"Trajectory count ({num_trajectories}) not divisible by VLM count ({num_vlms}). "
                f"Discarding last {remainder} trajectories."
            )
            trajectory_list = trajectory_list[:-remainder]

        # -------------------------------------------unpack and collate the trajectories

        trajectories = [tup[0] for tup in trajectory_list]
        traj_batch = collate_trajectories(trajectories)
        model_inputs = [(tup[1],tup[2]) for tup in trajectory_list]

        # ---------------------------------- compute gae ----------------------------------------------
        print("Computing GAE...")
        config = bootstrapper.resolved_dict['training']['rl_config']

        # Note: compute_gae expects (B, T) inputs and returns (B, T)
        advantages, returns = advantage_estimator_fn(
            token_level_rewards=traj_batch['rewards'],
            # values=traj_batch['values'],
            response_mask=traj_batch['response_mask'],
            config = cfg.training.rl_config,
            # gamma=config.get('gamma', 0.99), # Fallback defaults if not in config
            # lam=config.get('lam', 0.95)
        )

        traj_batch['advantages'] = advantages
        traj_batch['returns'] = returns
        print(advantages.shape)
        print(f"Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")

        # ------------------------------ dispatch training ----------------------------
        logger.info("Starting training")
        future_metadata = {}
        training_futures = []
        perm_indices = np.random.permutation(len(trajectory_list)) # shuffle
        for batch_start in range(0, len(trajectory_list), num_vlms):
            
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

        print(f"Dispatched {len(training_futures)} training tasks.")
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
                        # .max() is safe if success is 00001 (sparse) or 11111 (broadcasted)
                        "rollout/success": traj_stats['success'].max().item(), 
                        "rollout/spl": traj_stats['spl'].max().item(),
                        "rollout/ep_rtn": traj_stats['returns'].mean().item(),
                    }
                    result |= rollout_stats
                    wandb_actor.log.remote(result)
                    completed_count += 1
                    
                    # Formatting: Assuming 'result' is a dict of metrics (loss, kl, etc.)
                    # Adjust keys based on your specific trainer return signature
                    print(f"[{completed_count}/{total_tasks}] Complete. {result}")
                    
                except Exception as e:
                    logger.error(f"[{completed_count}/{total_tasks}] Task failed: {e}")

finally:
    for trainer in trainers:
        ray.kill(trainer)
    for sim in sims:
        ray.kill(sim)
    if wandb_actor is not None:
        ray.kill(wandb_actor)
    ray.shutdown()