import ray
import os
from omegaconf import OmegaConf
from config_schema import *

# Use these imports for type hinting
from config_schema import VLMConfig, RolloutConfig, ResourceConfig, HabitatConfig, RunConfig
from typing import List, Dict, Any, Iterator, Optional,Union
import logging
import json

def save_hydra_config(config, save_dir: str, filename: str = "config.yaml"):
    """
    Saves a Hydra/OmegaConf object to a YAML file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    OmegaConf.save(config=config, f=save_path)
    print(f"📄 Config saved to: {save_path}")

def load_hydra_config(load_dir: str, filename: str = "config.yaml"):
    """
    Loads a Hydra/OmegaConf object from a YAML file.
    Returns None if file is missing.
    """
    load_path = os.path.join(load_dir, filename)
    if not os.path.exists(load_path):
        return None
        
    try:
        conf = OmegaConf.load(load_path)
        return conf
    except Exception as e:
        print(f"⚠️ Failed to load config from {load_path}: {e}")
        return None

def get_base_model(checkpoint):
    config_path = os.path.join(checkpoint, "adapter_config.json")
    saved_base = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = json.load(f)
            saved_base = conf.get("base_model_name_or_path", "")
    return saved_base

def resolve_checkpoint_path(path_or_id):
    """
    Detects if the input is a local path or a HuggingFace Hub ID.
    If Hub ID, downloads the snapshot (adapters + optim states) and returns local cache path.
    If local path, returns as-is.
    """
    import os
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    # 1. If it exists locally, trust it.
    if os.path.exists(path_or_id):
        return path_or_id

    # 2. Heuristic: Hub IDs usually look like 'user/repo'
    # We attempt to download. If it fails, we assume it was a bad local path.
    print(f"🔍 '{path_or_id}' not found locally. Attempting HF Hub download...")
    
    try:
        # Download everything needed for a resume:
        # - Adapter weights (.bin or .safetensors)
        # - Configs (.json, .yaml)
        # - Optimizer/Scheduler states (.pt) - ONLY if you uploaded them!
        local_dir = snapshot_download(
            repo_id=path_or_id,
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.pt", "*.yaml"],
            ignore_patterns=["*.msgpack", "*.h5"], # Ignore flax/tf weights if any
            tqdm_class=None # Optional: silence progress bar
        )
        print(f"✅ Downloaded '{path_or_id}' to: {local_dir}")
        return local_dir
        
    except (RepositoryNotFoundError, RevisionNotFoundError):
        print(f"❌ Could not find '{path_or_id}' on HuggingFace Hub or locally.")
        raise FileNotFoundError(f"Checkpoint path not found: {path_or_id}")
    except Exception as e:
        print(f"⚠️ Error during HF download: {e}")
        raise e
    
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
        env_dict = None
        if res_cfg.vlm_conda_env is not None:
            env_dict = {"conda": res_cfg.vlm_conda_env}
        # We use the dicts directly to avoid pickling issues
        RemoteRLWorker = ray.remote(RLRayWorker).options(
            resources={res_cfg.vlm_resource_tag: 1},
            num_cpus=res_cfg.vlm_cpus,
            num_gpus=res_cfg.vlm_gpu_fraction,
            runtime_env=env_dict
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
        if res_cfg.master_port is None:
            import socket

            def find_free_port():
                # Create a new socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # Bind to an empty address/port, letting the OS pick a free port (port 0)
                    s.bind(('', 0))
                    # Return the port number assigned by the OS
                    return s.getsockname()[1]

            # Usage:
            master_port = find_free_port()
        else:
            master_port = res_cfg.master_port
        for rank, w in enumerate(workers):
            futures.append(w.setup_training.remote(
                config=train_cfg,
                rank=rank,
                world_size=world_size,
                master_addr=res_cfg.master_addr,
                master_port=master_port,
            ))
       
        return futures

class SimWorkerFactory:
    @staticmethod
    def create(sim_dict: dict, res_cfg: ResourceConfig, task_cfg: RunConfig, logger_actor=None):
        from utils.inference_core import HabitatRayWorker
        env_dict = None
        if res_cfg.habitat_conda_env is not None:
            env_dict = {"conda": res_cfg.habitat_conda_env}
        RemoteSim = ray.remote(HabitatRayWorker).options(
            resources={res_cfg.sim_resource_tag: 1},
            num_cpus=res_cfg.sim_cpus,
            num_gpus=res_cfg.sim_gpu_fraction,
            runtime_env=env_dict
        )

        handles = []
        for i in range(res_cfg.num_sims):
            # Calculate dynamic per-worker arguments
            log_dir = os.path.join(task_cfg.output_dir, task_cfg.run_name,"rollout")# f'worker_{i}')
            
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
                    res.sim_resource_tag: res.num_sims,
                },
                ignore_reinit_error=True,
                object_spilling_directory = res.object_spilling_directory
            )
        else:
            ray.init(address=res.ray_address, ignore_reinit_error=True)
    def bootstrap_logger(self):
        save_hydra_config(self.typed_cfg,os.path.join(self.typed_cfg.task.output_dir,self.typed_cfg.task.run_name))
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
    
    def bootstrap_vlms_rl(self,training=True):
        if self.typed_cfg.training.checkpoint is not None:
            checkpoint_path = self.typed_cfg.training.checkpoint
            checkpoint_path = resolve_checkpoint_path(checkpoint_path)
            base_model_path = get_base_model(checkpoint_path)
            if base_model_path is not None:
                self.resolved_dict['vlm']['model_id'] = base_model_path
                self.typed_cfg.vlm.model_id = base_model_path
            # self.resolved_dict['training']['checkpoint'] = checkpoint_path
            self.typed_cfg.training.checkpoint = checkpoint_path
        workers = RLWorkerFactory.create(
            vlm_dict=self.resolved_dict['vlm'], 
            rollout_dict=self.resolved_dict['rollout'], 
            res_cfg=self.typed_cfg.resources,
        )
        if training:
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
    
def trivial_shard_iterator(n=256) -> Iterator[None]:
    """Yields the trivial shard (None) once. Habitat handles dataset loading."""
    for i in range(n):
        yield None

def chunk_list(all_episodes: List[str], shard_size: int) -> Iterator[List[str]]:
    """Yields chunks of episodes of a specific size."""
    for i in range(0, len(all_episodes), shard_size):
        yield all_episodes[i : i + shard_size]
        
def get_console_logger(name = "DriverMain"):
    """Sets up a central logger and directory structure."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def get_shard_iterator(
    shard_size: int, 
    subset_label: str = "", 
    episode_json: str = "", logger: Optional[logging.Logger] = None,
    excluded_episodes = None
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
    if excluded_episodes is not None:
        excluded_episodes = set(excluded_episodes)
        all_episodes = [episode for episode in all_episodes if episode not in excluded_episodes]
    return chunk_list(all_episodes, shard_size)