import ray
import os
from omegaconf import OmegaConf
from config_schema import *

# Use these imports for type hinting
from config_schema import VLMConfig, RolloutConfig, ResourceConfig, HabitatConfig, RunConfig
from typing import List, Dict, Any, Iterator, Optional,Union
import logging
import json

class InferenceWorkerFactory:
    @staticmethod
    def create(vlm_dict: dict, rollout_dict: dict, res_cfg: ResourceConfig):
        # res_cfg is fine to keep as object for resource logic
        from utils.inference_core import InferenceRayWorker
        
        # We use the dicts directly to avoid pickling issues
        RemoteInferenceWorker = ray.remote(InferenceRayWorker).options(
            resources={res_cfg.vlm_resource_tag: 1},
            num_cpus=res_cfg.vlm_cpus,
            num_gpus=res_cfg.vlm_gpu_fraction,
            runtime_env={"conda": res_cfg.vlm_conda_env}
        )

        return [
            RemoteInferenceWorker.remote(
                rollout_config=rollout_dict, 
                **vlm_dict
            ) for _ in range(res_cfg.num_vlms)
        ]
    
class RLWorkerFactory:
    @staticmethod
    def create(vlm_dict: dict, rollout_dict: dict, res_cfg: ResourceConfig):
        # res_cfg is fine to keep as object for resource logic
        from utils.inference_core import RLRayWorker
        
        # We use the dicts directly to avoid pickling issues
        RemoteRLWorker = ray.remote(RLRayWorker).options(
            resources={res_cfg.vlm_resource_tag: 1},
            num_cpus=res_cfg.vlm_cpus,
            num_gpus=res_cfg.vlm_gpu_fraction,
            runtime_env={"conda": res_cfg.vlm_conda_env}
        )
        workers =  [
            RemoteRLWorker.remote(
                rollout_config=rollout_dict, 
                **vlm_dict
            ) for _ in range(res_cfg.num_vlms)
        ]
        
        return workers
    def _enable_training(workers,res_cfg:ResourceConfig,train_cfg:VLMTrainingConfig):
        # Auto-detect rendezvous point for the workers
        world_size = len(workers)        
        futures = []
        for rank, w in enumerate(workers):
            futures.append(w.setup_training.remote(
                config=train_cfg,
                rank=rank,
                world_size=world_size,
                master_addr=res_cfg.master_addr,
                master_port=res_cfg.master_port,
            ))
       
        return futures

class SimWorkerFactory:
    @staticmethod
    def create(sim_dict: dict, res_cfg: ResourceConfig, task_cfg: RunConfig, logger_actor=None):
        from utils.inference_core import HabitatRayWorker
        
        RemoteSim = ray.remote(HabitatRayWorker).options(
            resources={res_cfg.sim_resource_tag: 1},
            num_cpus=res_cfg.sim_cpus,
            num_gpus=res_cfg.sim_gpu_fraction,
            runtime_env={"conda": res_cfg.habitat_conda_env}
        )

        handles = []
        for i in range(res_cfg.num_sims):
            # Calculate dynamic per-worker arguments
            log_dir = os.path.join(task_cfg.output_dir, task_cfg.run_name, f'worker_{i}')
            
            # We merge the static sim_dict with our dynamic arguments
            h = RemoteSim.remote(
                **sim_dict,
                logging_output_dir=log_dir,
                logger_actor=logger_actor,
                # Ensure these match your HabitatRayWorker __init__
            )
            handles.append(h)
        return handles

class WandbFactory:
    @staticmethod
    def create(run_cfg: RunConfig, res_cfg: ResourceConfig, full_dict_cfg: dict):
        if not run_cfg.wandb_project: 
            return None
        
        from utils.logging_workers import WandbLoggerActor
        
        RemoteLogger = ray.remote(WandbLoggerActor).options(
            num_cpus=0, 
            runtime_env={"conda": res_cfg.vlm_conda_env}
        )

        return RemoteLogger.remote(
            wandb_init_kwargs={
                "project": run_cfg.wandb_project,
                "name": run_cfg.run_name,
                "job_type": "eval" # Hardcode or add to RunConfig schema
            },
            run_config=full_dict_cfg
        )

class ExpBootstrapper:
    def __init__(self, cfg: Union[InferenceConfig,RLConfig]):
        # Resolve all interpolations (Stage 1)
        # This turns ${read_text:...} into actual file content
        self.resolved_dict = OmegaConf.to_container(cfg, resolve=True)
        self.typed_cfg = cfg 

    def setup_cluster(self):
        res = self.typed_cfg.resources
        if res.ray_address == "local":
            ray.init(
                resources={
                    res.vlm_resource_tag: res.num_vlms, 
                    res.sim_resource_tag: res.num_sims
                },
                ignore_reinit_error=True
            )
        else:
            ray.init(address=res.ray_address, ignore_reinit_error=True)
    def bootstrap_logger(self):
        return WandbFactory.create(
            self.typed_cfg.task, 
            self.typed_cfg.resources, 
            self.resolved_dict
        )
    
    def bootstrap_vlms_infer(self):
        return InferenceWorkerFactory.create(
            vlm_dict=self.resolved_dict['vlm'], 
            rollout_dict=self.resolved_dict['rollout'], 
            res_cfg=self.typed_cfg.resources
        )
    
    def bootstrap_vlms_rl(self):
        workers = RLWorkerFactory.create(
            vlm_dict=self.resolved_dict['vlm'], 
            rollout_dict=self.resolved_dict['rollout'], 
            res_cfg=self.typed_cfg.resources,
        )
        futures = RLWorkerFactory._enable_training(workers,self.typed_cfg.resources,self.typed_cfg.training)
        ray.get(futures)
        return workers
    
    def bootstrap_sims(self,logger=None):
        return SimWorkerFactory.create(
            sim_dict=self.resolved_dict['sim'], 
            res_cfg=self.typed_cfg.resources, 
            task_cfg=self.typed_cfg.task, 
            logger_actor=logger
        )
    
    def bootstrap_eval(self):
        self.setup_cluster()
        
        # 1. Spawn Logger (Pass the FULL resolved dict for WandB hyperparams)
        logger = self.bootstrap_logger()
        
        # 2. Spawn Inference Workers
        # We pass the resolved dictionaries from our resolved_dict
        vlms = self.bootstrap_vlms_infer()
        
        # 3. Spawn Sim Workers
        sims = self.bootstrap_sims(logger)
        
        return vlms, sims, logger
    
def trivial_shard_iterator() -> Iterator[None]:
    """Yields the trivial shard (None) once. Habitat handles dataset loading."""
    yield None

def chunk_list(all_episodes: List[str], shard_size: int) -> Iterator[List[str]]:
    """Yields chunks of episodes of a specific size."""
    for i in range(0, len(all_episodes), shard_size):
        yield all_episodes[i : i + shard_size]
        
def get_console_logger():
    """Sets up a central logger and directory structure."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("EvalMain")

def get_shard_iterator(
    shard_size: int, 
    subset_label: str = "", 
    episode_json: str = "", logger: Optional[logging.Logger] = None
) -> Iterator[Optional[List[str]]]:
    """
    Orchestrates shard creation based on config.
    Reproduces the original branching logic for trivial vs. explicit shards.
    """
    # Case A: Trivial Shard (Let Habitat handle loading via its own config)
    if shard_size <= 0:
        logger.info("Using trivial shard (full dataset via Habitat config).")
        return trivial_shard_iterator()

    # Case B: Explicit Sharding (We must load the list first)
    all_episodes = []

    if subset_label:
        # Import inside function to avoid circular dependencies or heavy startup
        from constants import episode_labels_table
        if subset_label in episode_labels_table:
            all_episodes = episode_labels_table[subset_label]
            logger.info(f"Loaded {len(all_episodes)} episodes from subset: {subset_label}")
        else:
            raise ValueError(f"Subset label '{subset_label}' not found in constants.")

    elif episode_json:
        with open(episode_json, 'r') as f:
            all_episodes = json.load(f)
        logger.info(f"Loaded {len(all_episodes)} episodes from JSON: {episode_json}")

    else:
        raise ValueError("Shard size > 0 but no episode source (subset_label or episode_json) provided.")

    if not all_episodes:
        raise ValueError("The resolved episode list is empty.")

    return chunk_list(all_episodes, shard_size)