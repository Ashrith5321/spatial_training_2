from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# --- 1. Resource & Environment Config ---
@dataclass
class ResourceConfig:
    ray_address: str = "local"
    object_spilling_directory: str = "./ray_object_spilling"
    vlm_resource_tag: str = "env_a"
    sim_resource_tag: str = "env_b"
    master_addr: str = 'localhost'
    master_port: Optional[int] = None #port for accelerate/ddp
    num_vlms: int = 1
    num_sims: int = 1
    vlm_conda_env: Optional[str] = "vlm_node_1016"
    habitat_conda_env: Optional[str] = "vln"
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

    # clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    # cliprange = config.clip_ratio
    # cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    # cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    # clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    # clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0
@dataclass 
class PolicyLossConfig:
    clip_cov_ratio: Optional[float] = 0.0002
    clip_cov_ub: Optional[float] = 5.0
    clip_cov_lb: Optional[float] = 1.0
    
@dataclass 
class RLAlgoConfig:
    # generic on policy params
    use_value: bool = False
    value_grad_scale: float = 0.1
    advantage_estimator: str = "gae"
    policy_loss_name: str = "vanilla"
    n_rollout: int = 12 # note: must be divisible by num vlms times gradient accumulation
    n_adv: int = 256 # number of trajectories for advantage estimation, must > n_rollout
    n_epoch: int = 2 # number of policy gradient epochs

    # PPO Hyperparameters
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    clip_ratio_c: float = 3.0
    loss_agg_mode: Optional[str] = "seq-mean-token-sum"
    
    # clip cov parameters
    policy_loss:PolicyLossConfig = field(default_factory=PolicyLossConfig)
    # GAE Hyperparameters
    gamma: float = 0.99
    lam: float = 0.95

    time_kernel_sigma: float = 5.0
    distance_kernel_sigma: float = 0.5

    # Value & Entropy
    cliprange_value: float = 0.2
    entropy_bonus: float = 0.0

    # Ref KL Control
    use_ref: bool = True
    kl_coeff: float = 0.001
    kl_target: float = 0.1

    # # Compatibility for verl's agg_loss
    # @property
    # def global_batch_info(self):
    #     # For single-worker testing, batch size is 1
    #     return {}# "dp_size": 1, "global_batch_size": 1
    global_batch_info: Optional[Dict[str,Any]] = field(default_factory=lambda:{})
    # Helper to support config.get("key", default) used in loss functions
    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class SFTConfig:
    pass

# --- training configs ---
@dataclass
class HydraLoraConfig:
    """
    A Hydra-compatible mirror of peft.LoraConfig.
    Removes Union types (like str | List[str]) that crash OmegaConf.
    """
    r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Enforce List[str] to satisfy Hydra. 
    # If you need regex (str), you can change this to Any, but List is safer.
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # modules_to_save is also a list, defaulting to None is fine for Hydra
    modules_to_save: Optional[List[str]] = None

@dataclass
class VLMTrainingConfig:
    # checkpoints
    checkpoint:Optional[str] = None
    load_optim:bool = False # l
    load_sched:bool = False

    # Optimization
    learning_rate: float = 5e-6
    grad_accum_steps: int = 1
    mixed_precision: Optional[str] = "no" #['no', 'fp8', 'fp16', 'bf16']
    gradient_checkpointing: bool = True
    total_optimization_steps: int = 100000 # used for linear LR schedule
    warmup_steps: int = 64
    save_step: Optional[int] = 10

    # Value Head Configuration
    value_head_learning_rate: float = 5e-4  # Often higher than Adapter LR
    value_head_dropout: float = 0.0
    value_head_dtype: str = "float32"  
    # List of hidden layer sizes. Empty list [] implies a single linear layer (Linear Probe).
    value_head_hidden_dims: List[int] = field(default_factory=lambda:[1024,512])

    # PEFT: Pass the actual configuration object here (e.g., LoraConfig)
    # Typed as Any to avoid crashing if peft isn't installed on the driver
    peft_config: Optional[Any] = field(default_factory=HydraLoraConfig) 

    rl_config:Optional[RLAlgoConfig] = field(default_factory=RLAlgoConfig) # RL Algorithm 
    sft_config:Optional[SFTConfig] = None

# --- habitat sim configs ---
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
        "info": {"episode_label": True, "spl": True, "soft_spl":True, "success": True,"distance_to_goal":True},
        "done": True,
        "reward": True,
        "stuck": True,
        "fp_stop": True
    })
    auto_flush: bool = True # automatically flush logs upon reset
    ep_seed: Optional[bool] = None # if set, episode iterators are deterministic with same set seed all habitat workers

# --- Rollouts (both for Eval and RL) ---
@dataclass
class RolloutConfig:
    max_steps: int = 300
    temperature: float = 1.0
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

# --- Experiment housekeeping ---
@dataclass
class RunConfig:
    run_name: str = "debug_run"
    wandb_project: Optional[str] = None
    shard_size: int = 6
    subset_label: str = "sample400_a"
    episode_json: str = ""
    output_dir: str = "./dump/results"
    jobtype: str = "eval"

# --- ROOT CONFIGs ---
@dataclass
class InferenceConfig:
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sim: HabitatConfig = field(default_factory=HabitatConfig)
    task: RunConfig = field(default_factory=RunConfig)

@dataclass
class RLConfig(InferenceConfig):
    training: VLMTrainingConfig = field(default_factory=VLMTrainingConfig)