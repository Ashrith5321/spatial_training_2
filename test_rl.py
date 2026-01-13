import os 
os.environ['CUDA_VISIBLE_DEVICES']='1,2'
import hydra
from hydra import compose, initialize
from conf.register_configs import register_configs
from utils.factories import ExpBootstrapper,RLWorkerFactory
from utils.inference_core import run_inference_driver, create_shard_iterator
import ray
from utils.tensor_utils import TensorPacker
from utils.rl_core import collate_trajectories
import torch
import numpy as np

# Placeholder import for verl GAE
try:
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
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
        "task.run_name=test",
        "rollout.max_steps=20",
        "vlm.save_outputs=True", # need this for RL
        "resources.num_vlms=2",
        "resources.num_sims=2",
    ])

print(f"Model ID: {cfg.vlm.model_id}")
bootstrapper = ExpBootstrapper(cfg)

bootstrapper.setup_cluster()
# bootstrapper.bootstrap_all()

sims = bootstrapper.bootstrap_sims()
trainers = bootstrapper.bootstrap_vlms_rl()
shard_futures = [sim.assign_shard.remote(None) for sim in sims]
ray.get(shard_futures)

trajectory_list = []
for i in range(4): #collect 8 rollouts
    episode_futures = [trainer.run_episode.remote(sim,ray.get(sim.reset.remote())) for trainer,sim in zip(trainers,sims)]
    episode_outcomes = ray.get(episode_futures)

    trajectory_futures = [trainer.postprocess_episode.remote() for trainer in trainers]
    trajectory_list+=ray.get(trajectory_futures) #list of tuples (trajectory,model_inputs,metadata)

trajectories = [tup[0] for tup in trajectory_list]
traj_batch = collate_trajectories(trajectories)
print("Computing GAE...")
config = bootstrapper.resolved_dict['rollout'] # Or worker.rl_algo_config depending on where you stored gamma/lam

# Note: compute_gae expects (B, T) inputs and returns (B, T)
advantages, returns = compute_gae_advantage_return(
    token_level_rewards=traj_batch['rewards'],
    values=traj_batch['values'],
    response_mask=traj_batch['response_mask'],
    gamma=config.get('gamma', 0.99), # Fallback defaults if not in config
    lam=config.get('lam', 0.95)
)

traj_batch['advantages'] = advantages
traj_batch['returns'] = returns
print(advantages.shape)
print(f"Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")
model_inputs = [(tup[1],tup[2]) for tup in trajectory_list]

training_futures = [trainer.train_rl_step.remote(*model_inputs[idx],traj_batch[idx:(idx+1),traj_batch['response_mask'][idx]]) for idx,trainer in enumerate(trainers)]
print(ray.get(training_futures))