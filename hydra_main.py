import hydra
from config_schema import InferenceConfig
from conf.register_configs import register_configs

# 1. Register our command variants
register_configs()

@hydra.main(version_base=None, config_name="inference_config")
def main(cfg: InferenceConfig):
    import json
    import os
    from utils.factories import InferenceBootstrapper,get_shard_iterator,get_console_logger
    from utils.inference_core import run_inference_driver
    # 2. Setup Ray and Workers
    bootstrapper = InferenceBootstrapper(cfg)
    vlm_handles, sim_handles, logger_handle = bootstrapper.bootstrap_all()
    logger = get_console_logger()
    # 3. Prepare Data Shards (using simple helper)
    shard_iter = get_shard_iterator(
        subset_label= cfg.task.subset_label,
        episode_json= cfg.task.episode_json,
        shard_size=cfg.task.shard_size,
        logger=logger
    )

    # 4. Launch Driver
    try:
        metrics = run_inference_driver(
            sim_handles=sim_handles,
            vlm_handles=vlm_handles,
            shard_iterator=shard_iter
        )

        # 5. Save Results
        out_file = os.path.join(cfg.task.output_dir, cfg.task.run_name, "metrics.jsonl")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'a') as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")

    finally:
        if logger_handle:
            import time
            time.sleep(30) # wait for WandB
            ray.get(logger_handle.close.remote())
        import ray
        ray.shutdown()

if __name__ == "__main__":
    main()