import os
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from config_schema import *
# --- 1. The Resolver ---
def read_text_from_file(path: str) -> str:
    # Use get_original_cwd so it works even when Hydra shifts directories
    try:
        # If we are inside Hydra, use its original path
        base_path = get_original_cwd()
    except ValueError:
        # If we are running a unit test or script without Hydra
        base_path = os.getcwd()
        
    full_path = os.path.join(base_path, path)
    with open(full_path, 'r') as f:
        return f.read().strip()

# Register it immediately at the module level
# This ensures it's available as soon as this file is imported
OmegaConf.register_new_resolver("read_text", read_text_from_file, replace=True)

def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="inference_config", node=InferenceConfig)
    cs.store(name="rl_config",node=RLConfig)
    import conf.habitat_configs,conf.vlm_configs  # side effects (yikes)
    cs.store(name="l2", group="resources", node=ResourceConfig(num_sims=3,num_vlms=2)) #local 2 gpu
    cs.store(name="l3", group="resources", node=ResourceConfig(num_sims=4,num_vlms=3)) #local 2 gpu
    cs.store(name="l4", group="resources", node=ResourceConfig(num_sims=5,num_vlms=4)) #local 2 gpu