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
        "task.run_name=rpp_t",
        "task.wandb_project=rl_dev",
        "rollout.max_steps=300",
        "vlm.save_outputs=True", # need this for RL
        "task.subset_label=sample400",
        "sim.split=train_mini",
        "task.shard_size=0", # no sharding, fulldataset for everyone
        "resources.num_vlms=2",
        "resources.num_sims=3",
        "resources.master_port=25657",
        # f"training.rl_config.advantage_estimator={AdvantageEstimator.REINFORCE_PLUS_PLUS}",
        f"training.rl_config.advantage_estimator=reinforce_plus_plus_geometric_time_aware",
        "training.grad_accum_steps=2",
        "training.rl_config.gamma=0.95",
        "training.learning_rate=2e-5",
        "sim.fp_guard=false",
        "training.rl_config.use_value=false",
        "training.save_step=100"
        # "training.checkpoint=/Projects/spatial_training/test_checkpoint"
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

# shard_futures = [sim.assign_shard.remote(None) for sim in sims]
# ray.get(shard_futures)

#
# # 3. Prepare Data Shards (using simple helper)
shard_iter = get_shard_iterator(
    subset_label= cfg.task.subset_label,
    episode_json= cfg.task.episode_json,
    shard_size=cfg.task.shard_size,
    logger=logger
)
trajectory_list = []

N_EPISODES = 10000 #10k for train_mini, 2k for val
FREEZE_DATA = False # no longer debugging
try:
    for i in range(10000):
        if (i+1)*bootstrapper.typed_cfg.training.rl_config.n_rollout>N_EPISODES or FREEZE_DATA:
            # reset the dataset
            shard_iter = get_shard_iterator(
                subset_label= cfg.task.subset_label,
                episode_json= cfg.task.episode_json,
                shard_size=cfg.task.shard_size,
                logger=logger
            )
        # ------------------------------------------- rollouts ------------------------------------------
        logger.info("Starting rollout collection!")

        rollout_list = collect_rollouts(sims,trainers,shard_iter,target_episodes=bootstrapper.typed_cfg.training.rl_config.n_rollout) #

        print("done collecting")
        num_vlms = len(trainers)
        # -------------------------------------------unpack and collate the trajectories

        trajectory_list += [tup[0] for tup in rollout_list]
        trajectory_list = trajectory_list[-bootstrapper.typed_cfg.training.rl_config.n_adv:]
        traj_batch = collate_trajectories(trajectory_list)
        model_inputs = [(tup[1],tup[2]) for tup in rollout_list]

        # ---------------------------------- compute gae ----------------------------------------------
        print("Computing Advantages")
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
        traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout:] # only train on most recent.
        print(f"Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")

        # ------------------------------ non logged training (extra epochs) ----------
        logger.info("Starting training")

        for i in range(bootstrapper.typed_cfg.training.rl_config.n_epoch-1):
            print(f"epoch {i}")
            training_futures = []
            perm_indices = np.random.permutation(bootstrapper.typed_cfg.training.rl_config.n_rollout) # shuffle
            for batch_start in range(0, bootstrapper.typed_cfg.training.rl_config.n_rollout, num_vlms):
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
                training_futures.extend(step_futures)
            ray.get(training_futures)
        # ------------------------------ dispatch logged training ----------------------------
        future_metadata = {}
        training_futures = []
        perm_indices = np.random.permutation(bootstrapper.typed_cfg.training.rl_config.n_rollout) # shuffle
        for batch_start in range(0, bootstrapper.typed_cfg.training.rl_config.n_rollout, num_vlms):
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

        print(f"Dispatched {len(training_futures)} training tasks for final epoch")
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
        #------------------------------------ save checkpoint ------------------------------------
        if (i+1) % bootstrapper.typed_cfg.training.save_step == 0:
            print("saving checkpoint")
            ray.get(trainers[0].save_checkpoint_unsafe.remote(os.path.join(bootstrapper.typed_cfg.task.output_dir,bootstrapper.typed_cfg.task.run_name,"checkpoints",f"checkpoint_{i}")))

finally:
    for trainer in trainers:
        ray.kill(trainer)
    for sim in sims:
        ray.kill(sim)
    if wandb_actor is not None:
        ray.kill(wandb_actor)
    ray.shutdown()