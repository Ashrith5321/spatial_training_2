from hydra.core.config_store import ConfigStore
from longnav.config_schema import HabitatConfig
cs = ConfigStore.instance()

cs.store(name="voxel",group="sim", node=HabitatConfig(
    voxel_kwargs={
        "patch_size": 32,
        "resolution": 0.15,
        "fov_degrees": 79
    }, #set to none for standard mode
    output_schema={
        "obs": {"rgb": True, "instr_or_goal": True, "patch_coords": False},
        "info": {"episode_label": True, "spl": True, "success": True},
        "done": True,
    }
))