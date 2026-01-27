'''
🚀 [run experiment]:
python3 training_scripts/train_rl.py +experiment=<experiment_name> +node=<goldeen/lighthouse>
NOTE: experiment_name must be a config that exists in conf/experiment.

⚙️ [add experiment config]:
add new yaml to conf/experiment. see config_schema.py for requirements or reference existing yaml.
NOTE: need to have "# @package _global_" at the start of your config.

👾 [see hydra help]:
python3 training_scripts/train_rl.py --help
https://hydra.cc/docs/intro/

🔧 [install tab completion]:
eval "$(python training_scripts/train_rl.py -sc install=bash)"
NOTE: tab completion only works if your command uses python not python3. somehow.
'''
import os
# NUCLEAR THREAD CAP: Must be set before importing numpy/torch/ray
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import hydra
from conf.register_configs import register_configs
from config_schema import RLConfig
import os 

DEBUG_FLAG = False
FREEZE_DATA = False # for debugging only

# 1. Register our command variants
register_configs()
@hydra.main(version_base=None, config_name="rl_config",config_path='../conf')
def main(cfg: RLConfig):
    # keep heavy imports here so hydra tab complete is snappier?
    import ray
    import numpy as np

    from conf.register_configs import register_configs
    from utils.factories import ExpBootstrapper,get_shard_iterator,get_console_logger
    from utils.tensor_utils import TensorPacker
    from utils.rl_core import collate_trajectories,collect_rollouts
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn,AdvantageEstimator,POLICY_LOSS_REGISTRY
    import signal
    import json

    def debug_signal_handler(sig, frame):
        # should allow us to interrupt the loop, save data etc, and resume
        global DEBUG_FLAG
        DEBUG_FLAG = True
        decision = input("debug: y, exit: n, wait: any other key")
        if decision == 'y':
            import ipdb
            ipdb.set_trace()
        if decision == 'n':
            try:
                cleanup()
            finally:
                sys.exit()
    # signal.signal(signal.SIGINT, debug_signal_handler)


    advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)
    print(f"Model ID: {cfg.vlm.model_id}")
    bootstrapper = ExpBootstrapper(cfg)
    logger = get_console_logger()

    bootstrapper.setup_cluster()
    trainers = bootstrapper.bootstrap_vlms_rl() #allocate vlms first to prevent out of room issues
    wandb_actor = bootstrapper.bootstrap_logger()
    sim_logger = None
    sims = bootstrapper.bootstrap_sims(sim_logger)

    # # 3. Prepare Data Shards (using simple helper)
    shard_iter = get_shard_iterator(
        subset_label= cfg.task.subset_label,
        episode_json= cfg.task.episode_json,
        shard_size=cfg.task.shard_size,
        logger=logger
    )
    trajectory_list = []

    def cleanup():
        for trainer in trainers:
            ray.kill(trainer)
        for sim in sims:
            ray.kill(sim)
        if wandb_actor is not None:
            ray.kill(wandb_actor)
        ray.shutdown()

    def debug():
        global DEBUG_FLAG
        DEBUG_FLAG = False # consume the flag
        import ipdb
        ipdb.set_trace()

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

    try:
        shard_iter = get_shard_iterator(
            subset_label= cfg.task.subset_label,
            episode_json= cfg.task.episode_json,
            shard_size=cfg.task.shard_size,
            logger=logger
        )
        num_rollouts = bootstrapper.typed_cfg.training.total_optimization_steps*bootstrapper.typed_cfg.training.grad_accum_steps//bootstrapper.typed_cfg.training.rl_config.n_rollout
        for global_cycle in range(num_rollouts):
            if FREEZE_DATA:
                # reset the dataset
                shard_iter = get_shard_iterator(
                    subset_label= cfg.task.subset_label,
                    episode_json= cfg.task.episode_json,
                    shard_size=cfg.task.shard_size,
                    logger=logger
                )
            # ------------------------------------------- rollouts ------------------------------------------
            logger.info("Starting rollout collection!")

            # rollout_list = collect_rollouts(sims,trainers,shard_iter,target_episodes=bootstrapper.typed_cfg.training.rl_config.n_rollout) #
            rollout_list,result_list,log_list = collect_rollouts(sims,trainers,shard_iter,bootstrapper.typed_cfg.training.rl_config.n_rollout) #

            print("done collecting")
            num_vlms = len(trainers)
            # -------------------------------------------unpack and collate the trajectories

            trajectory_list += [tup[0] for tup in rollout_list]
            trajectory_list = trajectory_list[-bootstrapper.typed_cfg.training.rl_config.n_adv:]
            traj_batch = collate_trajectories(trajectory_list)
            model_inputs = [(tup[1],tup[2]) for tup in rollout_list]
            
            values = traj_batch.get("values",None)
            distances = traj_batch.get('distance_to_goal',None)
            # ---------------------------------- compute gae ----------------------------------------------
            print("Computing Advantages")
            # config = bootstrapper.resolved_dict['training']['rl_config']
            # Note: compute_gae expects (B, T) inputs and returns (B, T)
            adv_tuple = advantage_estimator_fn(
                token_level_rewards=traj_batch['rewards'],
                values=values,
                distances = distances,
                response_mask=traj_batch['response_mask'],
                config = cfg.training.rl_config,
                # gamma=config.get('gamma', 0.99), # Fallback defaults if not in config
                # lam=config.get('lam', 0.95)
            )
            advantages, returns = adv_tuple[0],adv_tuple[1]
            if len(adv_tuple)>2:
                traj_batch['baseline'] = adv_tuple[2]
                print("DEBUG: computing variances")
                print(f"Rtn Var: {(returns[traj_batch['response_mask']==1]).var().item():.4f}")
                print(f"MSE Error: {((traj_batch['baseline'][traj_batch['response_mask']==1]-returns[traj_batch['response_mask']==1])**2).mean().item():.4f}")
            traj_batch['advantages'] = advantages
            traj_batch['returns'] = returns
            global_return_mean = returns[traj_batch['response_mask']==1].mean().item()
            traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout:] # only train on most recent.
            print(f"Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")

            if DEBUG_FLAG:
                debug() # great spot to intercept the trajectories for saving etc
                
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
                            "rollout/success": traj_stats['success'].max().item(), 
                            "rollout/spl": traj_stats['spl'].max().item(),
                            "rollout/ep_rtn": traj_stats['returns'].mean().item(),
                            "rollout/rtn_var": traj_stats['returns'].var().item(),
                            "rollout/global_cycle": global_cycle
                        }
                        try:
                            critic_mse = ((traj_stats['baseline']-traj_stats['returns'])**2).mean()
                            naive_mse = ((traj_stats['returns'] - global_return_mean)**2).mean().item()
                            rollout_stats |= {
                                "rollout/baseline_mse":critic_mse,
                                "rollout/naive_mse":naive_mse
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
                            logger.warning(f"Log file for rollout {rollout_idx} not ready (Sim Worker I/O Lag). Skipping detailed logs.")
                        except Exception as e:
                            logger.warning(f"Failed to read log file: {e}")
                        wandb_actor.log_row.remote(result)
                        completed_count += 1
                     
                        print(f"[{completed_count}/{total_tasks}] Complete. {result}")
                        
                    except Exception as e:
                        logger.error(f"[{completed_count}/{total_tasks}] Task failed: {e}")
            #------------------------------------ save checkpoint ------------------------------------
            steps_until_save = (global_cycle+1) % bootstrapper.typed_cfg.training.save_step
            if steps_until_save == 0:
                print("saving checkpoint")
                ray.get(trainers[0].save_checkpoint_unsafe.remote(os.path.join(bootstrapper.typed_cfg.task.output_dir,bootstrapper.typed_cfg.task.run_name,"checkpoints",f"checkpoint_{global_cycle}")))
            else:
                print(f"T-{steps_until_save} steps until checkpoint!")
            print("saving debug...")
            pickle_obj(traj_batch, f"tb_{global_cycle}")
            pickle_obj(result_list, f"result_{global_cycle}")
            pickle_obj(ray.get(log_list),f"logpaths_{global_cycle}")
            del model_inputs
    finally:

        cleanup()

if __name__ == "__main__":
    main()