from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# --- 1. Resource & Environment Config ---
@dataclass
class ResourceConfig:
    ray_address: str = "local"
    vlm_resource_tag: str = "env_a"
    sim_resource_tag: str = "env_b"
    num_vlms: int = 1
    num_sims: int = 1
    vlm_conda_env: str = "vlm_node_1016"
    habitat_conda_env: str = "vln"
    vlm_gpu_fraction: float = 0.7
    sim_gpu_fraction: float = 0.14
    vlm_cpus: int = 4
    sim_cpus: int = 4

# --- 3. Model & Worker Configs ---
@dataclass
class VLMConfig:
    model_id: str = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
    attn_impl: str = "flash_attention_2"
    dtype: str = "bfloat16"
    prefix: str = '<|im_start|>assistant\n**'
    postfix: str = '**<|im_end|>\n'
    vocab: List[str] = field(default_factory=lambda: ["stop", "forward", "left", "right"])
    offload_cache: bool = False
    use_sparse: bool = True
    save_outputs: bool = False # only need this for RL

@dataclass
class HabitatConfig:
    config_path: str = "configs/objectnav_hm3d_rgbd_semantic.yaml"
    workspace: Optional[str] = "/Projects/SG_VLN_HumanData/SG-VLN"
    scenes_dir: Optional[str] = None
    split: str = "val"
    fp_guard: bool = True
    fn_guard: bool = False
    voxel_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: None)
    output_schema: Dict[str, Any] = field(default_factory=lambda: {
        "obs": {"rgb": True, "goal_name": True, "patch_coords": False},
        "info": {"episode_label": True, "spl": True, "success": True},
        "done": True,
        "reward": True,
        "stuck": True,
        "fp_stop": True
    })

@dataclass
class RolloutConfig:
    max_steps: int = 300
    temperature: float = 1.2
    action_space_str: str = "[stop, forward, left, right, up, down]"
    system_prompt: str = "${read_text:conf/prompts/objectnav_prompt.txt}"
    action_space: List[str] = field(default_factory=lambda: ["stop", "forward", "left", "right"])
    # Templates are lists of dicts (JSON-like)
    convo_start_template: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"role": "user", "content": [{"type": "text", "text": "${rollout.system_prompt}"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ])
    
    convo_turn_template: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"role": "assistant", "content": [{"type": "text", "text": "**$action**"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ])

# --- 4. Task / Rollout Logic ---
@dataclass
class RunConfig:
    run_name: str = "debug_run"
    wandb_project: Optional[str] = None
    shard_size: int = 6
    subset_label: str = "sample400_a"
    episode_json: str = ""
    output_dir: str = "./dump/results"
    jobtype: str = "eval"

# --- 5. THE MAIN ROOT CONFIG ---
@dataclass
class InferenceConfig:
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sim: HabitatConfig = field(default_factory=HabitatConfig)
    task: RunConfig = field(default_factory=RunConfig)