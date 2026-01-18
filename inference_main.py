from numpy import False_
import ray
import argparse
import logging
import json
import os
from typing import List, Iterator
import textwrap
# Import your workers and driver (assuming they are in `driver.py` or similar)
from utils.inference_core import run_inference_driver,VLMRayWorker,HabitatRayWorker
from utils.logging_workers import WandbLoggerActor
from utils.prompt_utils import SYSTEM_PROMPT_TEMPLATE, CONVO_TURN_TEMPLATE, CONVO_START_TEMPLATE

'''
Eval Usage Example
python3 inference_main.py --model-id Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800 --ray-address local --shard-size 6 --subset-label sample400_a --num-vlms 2 --num-sims 3 --attn-impl='flash_attention_2' --dtype='float16' --max-steps 300 --wandb-project 'single_action_eval' --run-name float16fa2
python3 inference_main.py --model-id Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800 --ray-address local --shard-size 6 --subset-label sample400_a --num-vlms 1 --num-sims 2 --attn-impl='flash_attention_2' --dtype='bfloat16' --max-steps 300 --wandb-project 'single_action_eval' --run-name bfloat16fa2_sparse

python3 inference_main.py --model-id Phyllis1/qwen3_sft_bev_test_20260102_060608_ckpt8700 --ray-address local --shard-size 6 --subset-label sample400_a --num-vlms 1 --num-sims 2 --attn-impl='flash_attention_2' --dtype='bfloat16' --max-steps 300 --wandb-project 'single_action_eval' --run-name bfloat16fa2_sparse_bev_8700
python3 inference_main.py --model-id Phyllis1/qwen3_sft_bev_rerun_26770828_ckpt6700 --ray-address local --shard-size 6 --subset-label sample400_a --num-vlms 1 --num-sims 2 --attn-impl='flash_attention_2' --dtype='bfloat16' --max-steps 300 --wandb-project 'single_action_eval' --run-name bfloat16fa2_sparse_bev_6700

'''
'''
Debug Usage Example

python3 inference_main.py --model-id Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800 --ray-address local --wandb-project 'test_mp' --shard-size 20 --subset-label sample400_a --num-vlms 1 --num-sims 2 --attn-impl='flash_attention_2' --dtype='bfloat16'
python3 inference_main.py --model-id Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800 --ray-address local --shard-size 0 --subset-label sample400_a --num-vlms 1 --num-sims 1 --attn-impl='flash_attention_2' --dtype='bfloat16' --max-steps 200
python3 inference_main.py --model-id Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800 --ray-address local --shard-size 30 --subset-label sample400_a --num-vlms 2 --num-sims 3 --attn-impl='flash_attention_2' --dtype='float16' --max-steps 300 --wandb-project 'test_mp'
'''
# If you have a dedicated logger actor
# from utils.logger import LoggingActor 
VLM_CONDA_ENV = "vlm_node_1016"      # Replace with your actual VLM conda env name
HABITAT_CONDA_ENV = "vln"

OUTPUT_SCHEMA = {
    "obs":{
        "rgb":True,
        "goal_name":True,
        "patch_coords":True,
        "step": True,
        "agent_loc": True,
        "agent_heading": True
    },
    "info":{
        "episode_label":True,
        "spl":True,
        "distance_to_goal":True,
        "soft_spl":True,
        "success":True,
    },
    "done":True,
}

# --- Logging Setup ---
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

# --- Sharding Helper ---
def create_shard_iterator(
    all_episodes: List[str], 
    shard_size: int
) -> Iterator[List[str]]:
    """
    Yields chunks of episodes.
    """
    # Simple list slicing generator
    for i in range(0, len(all_episodes), shard_size):
        yield all_episodes[i : i + shard_size]

def trivial_shard_iterator():
    '''
    yields the trivial shard (None) once.
    '''
    yield None

# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="Distributed Habitat-VLM Evaluation")
    
    # Cluster Config
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address")
    parser.add_argument("--num-vlms", type=int, default=1, help="Number of VLM workers")
    parser.add_argument("--num-sims", type=int, default=1, help="Number of Habitat workers")
    
    # Resource Isolation (Keys must match ray start --resources)
    parser.add_argument("--vlm-resource-tag", type=str, default="env_a", help="Ray resource tag for VLM env")
    parser.add_argument("--sim-resource-tag", type=str, default="env_b", help="Ray resource tag for Habitat env")

    # VLM Config
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct") #
    parser.add_argument("--attn-impl", type=str, default="sdpa")
    parser.add_argument("--dtype", type=str, default="float16")
    
    # Habitat Config
    parser.add_argument("--habitat-config", type=str, default="configs/objectnav_hm3d_rgbd_semantic.yaml")
    parser.add_argument("--habitat-workspace", type=str, default="/Projects/SG_VLN_HumanData/SG-VLN")
    parser.add_argument("--scenes-dir", type=str, help="Path to HM3D/Gibson scenes")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--debug_with_humanaction", action="store_true", help="Enable debugging mode using human action (defaults to False).")
    
    # Evaluation Config
    parser.add_argument("--episode-json", type=str, default="", help="Path to full list of episodes")
    parser.add_argument("--subset-label",type=str,default="")
    parser.add_argument("--shard-size", type=int, default=0, help="Episodes per shard")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="./dump/results")
    parser.add_argument("--run-name", type=str, default="eval_run_001")

    parser.add_argument("--wandb-project",type=str, default='')

    args = parser.parse_args()

    if args.debug_with_humanaction:
        args.habitat_config = "configs/objectnav_mp3d_rgbd_semantic_human.yaml"
        args.habitat_split = "val"
        args.subset_label = "humandata_20"

    # 1. Setup
    logger = get_console_logger()
    logger.info(f"Connecting to Ray at {args.ray_address}...")
    
    # Initialize Ray. 
    # If running locally for debug, we might want to simulate resources
    if args.ray_address == "local":
        ray.init(
            resources={
                args.vlm_resource_tag: args.num_vlms, 
                args.sim_resource_tag: args.num_sims
            },
            # log_to_driver=False
        )
    else:
        ray.init(address=args.ray_address)
    # action_space_list = ["stop","forward","left","right","up","down"]
    action_space_list = ["stop","forward","left","right"]
    experiment_config = {
        "rollout_config": {
            "max_steps": args.max_steps,
            "temperature": 1.0, #1.2, # Sweet spot
            "convo_start_template": CONVO_START_TEMPLATE, #template for the initial conversation chunk
            "convo_turn_template": CONVO_TURN_TEMPLATE, #template for the recurrent conversation chunks
            "action_space":action_space_list,
            "action_space_str":"[stop, forward, left, right, up, down]"
        },
        "vlm_config":{
            #prefix and postfixes to "sandwich" the decision token
            "prefix":'<|im_start|>assistant\n**',
            "postfix":'**<|im_end|>\n',
            "vocab":action_space_list, # restrict vocab to remove dangerous up/down actions
            "offload_cache":False, #offload the kv cache layers to save VRAM
            "use_sparse": True
        },
        "sim_config":{
            "fp_guard": True, # use oracle to prevent incorrect stop actions
            "fn_guard": False, # use oracle to automatically perform stop action

            # specifying this enables bev/patch voxel grid!
            # "voxel_kwargs" : {
            #     "patch_size":32,
            #     "resolution":0.15,
            #     "fov_degrees":79
            # }
        }
        
    }

    if args.wandb_project != "":
        RemoteWandbLogger = ray.remote(num_cpus=0,num_gpus=0,runtime_env={"conda": VLM_CONDA_ENV})(WandbLoggerActor)
        wandb_logger = RemoteWandbLogger.remote(wandb_init_kwargs={"project":args.wandb_project,"name":args.run_name,"job_type":'eval',"entity":"vlm_rl"},run_config = experiment_config)
    else:
        wandb_logger = None

    # 2. Prepare Data
    logger.info("Loading episode list...")
    
    # Initialize defaults
    shard_iter = None
    all_episodes = None

    if args.shard_size <= 0:
        # Case A: Trivial Shard (Let Habitat handle loading)
        shard_iter = trivial_shard_iterator()
        logger.info(f"Using the trivial shard (full dataset via Habitat config).")
    else:
        # Case B: Explicit Sharding (We must load the list first)
        if args.subset_label != "":
            from constants import episode_labels_table
            all_episodes = episode_labels_table[args.subset_label]
        elif args.episode_json != "":
            with open(args.episode_json, 'r') as f:
                all_episodes = json.load(f)
        else:
            logger.error("Shard size > 0 but no episode source provided (subset-label or json).")
            exit(1)

        try:
            shard_iter = create_shard_iterator(all_episodes, args.shard_size)
            logger.info(f"Loaded {len(all_episodes)} episodes. Creating shards of size {args.shard_size}.")
        except Exception as e:
            logger.error(f"Cannot create split shards: {e}")
            exit(1)

    # 3. Bootstrap Workers
    # Pass custom resource tags dynamically
    RemoteVLMWorker = VLMRayWorker.options(resources={args.vlm_resource_tag: 1},num_cpus=4,num_gpus=0.7,runtime_env=
            {"conda": VLM_CONDA_ENV,
                "env_vars":{
                # "CUDA_VISIBLE_DEVICES":"1"
                }
                # "env_vars": {
                #     "GLOG_minloglevel": "2",  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
                #     "MAGNUM_LOG": "quiet",    # Silences Magnum graphics engine
                #     "HABITAT_SIM_LOG": "quiet" # Silences Habitat Sim specific logs
                # }
            })
    RemoteHabitatWorker = HabitatRayWorker.options(resources={args.sim_resource_tag: 1},num_cpus=4,num_gpus=0.14,
        runtime_env={
            "conda": HABITAT_CONDA_ENV,
            "env_vars":{
                # "CUDA_VISIBLE_DEVICES":"2"
            }
        })
    experiment_config['vlm_config']['model_id'] = args.model_id
    experiment_config['vlm_config']['attn_implementation'] = args.attn_impl
    experiment_config['vlm_config']['dtype'] = args.dtype
                
    vlm_handles = [
        RemoteVLMWorker.remote(
            rollout_config=experiment_config['rollout_config'],
            debug_with_humanaction=args.debug_with_humanaction,
            **experiment_config['vlm_config']
        ) for _ in range(args.num_vlms)
    ]
    
    sim_handles = [
        RemoteHabitatWorker.remote(
            workspace=args.habitat_workspace,
            config_path=args.habitat_config,
            scenes_dir=args.scenes_dir,
            split=args.split,
            debug_with_humanaction=args.debug_with_humanaction,
            output_schema = OUTPUT_SCHEMA,
            logging_output_dir = str(os.path.join(args.output_dir, args.run_name, f'worker_{i}/')),
            logger_actor = wandb_logger,
            **experiment_config['sim_config']
            # enable_caching = False
        ) for i in range(args.num_sims)
    ]
    # 5. Launch Driver
    logger.info("Starting Inference Driver...")
    try:
        metrics = run_inference_driver(
            sim_handles=sim_handles,
            vlm_handles=vlm_handles,
            shard_iterator=shard_iter
        )
        # 6. Save Final Results
        out_file = os.path.join(args.output_dir, f"{args.run_name}","metrics.jsonl")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'a') as f:
            for metric in metrics:
                f.write(json.dumps(metric)+"\n")
            
        logger.info(f"Success! Metrics saved to {out_file}")
        
        if wandb_logger is not None:
            import time
            print("waiting 2 minutes for wandb logger to finalize!")
            time.sleep(120) #TODO: this is so dumb...
            ray.get(wandb_logger.close.remote())

    except KeyboardInterrupt:
        logger.warning("Job interrupted by user.")
    except Exception as e:
        logger.exception("Fatal error during inference driver execution.")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()

    

