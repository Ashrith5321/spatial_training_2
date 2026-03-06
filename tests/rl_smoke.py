from longnav.config_schema import *
from longnav.utils.factories import ExpBootstrapper,get_shard_iterator
from longnav.env.env_base import DummyEnvActor
from longnav.utils.rollout_core import collect_rollouts
from longnav.utils.rl_core import collate_trajectories
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

import ray
import numpy as np
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

bootstrapper = ExpBootstrapper(cfg)
bootstrapper.setup_cluster()

trainers = bootstrapper.bootstrap_vlms_rl(training=True) 
sims = [ray.remote(DummyEnvActor).remote() for _ in range(2)]
trajectory_list = []

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

print("training step")

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
    training_futures.extend(step_futures)
print(ray.get(training_futures))