'''
🚀 [run experiment]:
python3 rl_eval.py +experiment=eval_goldeen +training.checkpoint=<checkpoint_name_or_path>
NOTE: experiment_name must be a config that exists in conf/experiment.

⚙️ [add experiment config]:
add new yaml to conf/experiment. see config_schema.py for requirements or reference existing yaml.
NOTE: need to have "# @package _global_" at the start of your config.

👾 [see hydra help]:
python3 training_scripts/train_rl.py --help
https://hydra.cc/docs/intro/

🔧 [install tab completion]:
eval "$(python training_scripts/train_rl.py -sc install=bash)"
eval "$(python rl_eval.py -sc install=bash)"

NOTE: tab completion only works if your command uses python not python3. somehow.
'''

from pathlib import Path
import sys
import math
# _ROOT = Path(__file__).resolve().parents[1]
# if str(_ROOT) not in sys.path:
#     sys.path.insert(0, str(_ROOT))
import hydra
from conf.register_configs import register_configs
from config_schema import RLConfig
import os 
import itertools
DEBUG_FLAG = False

# 1. Register our command variants
register_configs()
@hydra.main(version_base=None, config_name="rl_config",config_path='./conf')
def main(cfg: RLConfig):
    # keep heavy imports here so hydra tab complete is snappier?
    import ray
    import numpy as np

    from conf.register_configs import register_configs
    from utils.factories import ExpBootstrapper,get_shard_iterator,get_console_logger
    from utils.tensor_utils import TensorPacker
    from utils.rl_core import collate_trajectories,collect_rollouts
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn,AdvantageEstimator,POLICY_LOSS_REGISTRY

    advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)
    print(f"Model ID: {cfg.vlm.model_id}")
    bootstrapper = ExpBootstrapper(cfg)
    logger = get_console_logger()

    bootstrapper.setup_cluster()
    trainers = bootstrapper.bootstrap_vlms_rl(training=False) #allocate vlms first to prevent out of room issues
    wandb_actor,episodes_to_skip = bootstrapper.bootstrap_logger()
    sim_logger = wandb_actor
    sims = bootstrapper.bootstrap_sims(sim_logger)

    # # 3. Prepare Data Shards (using simple helper)
    shard_iter = get_shard_iterator(
        subset_label= cfg.task.subset_label,
        episode_json= cfg.task.episode_json,
        shard_size=cfg.task.shard_size,
        logger=logger,
        excluded_episodes=episodes_to_skip
    )

    shard_iter,shard_iter_copy = itertools.tee(shard_iter)
    try:
        all_episodes = [s for shard in shard_iter_copy for s in shard]
    except:
        all_episodes = [None]*10000 #fallback

    def cleanup():
        for trainer in trainers:
            ray.kill(trainer)
        for sim in sims:
            ray.kill(sim)
        if wandb_actor is not None:
            ray.get(wandb_actor.close.remote())
            time.sleep(15)
            ray.kill(wandb_actor)
        ray.shutdown()

    # convenience functions for ipdb abuse
    def save_checkpoint(name):
        ray.get(trainers[0].save_checkpoint_unsafe.remote(os.path.join(bootstrapper.typed_cfg.task.output_dir,bootstrapper.typed_cfg.task.run_name,"checkpoints",f"manual_checkpoint_{name}")))

    def pickle_obj(obj,filename):
        import pickle
        dirname = os.path.join(bootstrapper.typed_cfg.task.output_dir,bootstrapper.typed_cfg.task.run_name,"dbg")
        os.makedirs(dirname,exist_ok=True)
        filepath = os.path.join(dirname,f"{filename}.pkl")
        with open(filepath,'wb') as f:
            pickle.dump(obj,f)
    
    # ------------------------------------------- rollouts ------------------------------------------
    batch_size = 32 # fixed batch size decoupled from RL logic for eval
    for i in range(max(math.ceil(len(all_episodes)/batch_size),1)):
        logger.info("Starting rollout collection!")
        # bootstrapper.typed_cfg.training.rl_config.n_rollout
        rollout_list,result_list,log_list = collect_rollouts(sims,trainers,shard_iter,batch_size,{"return_inputs":False,"eval":True}) #
        if len(rollout_list) == 0:
            print("rollout list empty, exiting")
            break
        # save for analysis
        # pickle_obj(rollout_list, f"rollout_{i}")
        # pickle_obj(result_list, f"result_{i}")
        # pickle_obj(log_list,f"logpaths_{i}")
    ray.get(log_list)
    import time
    # time.sleep(360)
    cleanup()

if __name__ == "__main__":
    main()