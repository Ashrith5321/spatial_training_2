from longnav.config_schema import *
from longnav.utils.factories import ExpBootstrapper,get_shard_iterator
from longnav.env.env_base import DummyEnvActor
from longnav.utils.rollout_core import collect_rollouts
import ray

cfg = RLConfig()
cfg.resources.osm_gb=28
cfg.resources.vlm_conda_env=None
cfg.vlm.attn_impl = "sdpa"
cfg.rollout.convo_start_template=[
        {"role": "user", "content": [{"type": "text", "text": "example substitution: $instr_or_goal"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ]

bootstrapper = ExpBootstrapper(cfg)
bootstrapper.setup_cluster()

vlms = bootstrapper.bootstrap_vlms_rl(training=False) 
sims = [ray.remote(DummyEnvActor).remote() for _ in range(2)]

rollout_list,result_list,log_list = collect_rollouts(sims,vlms,get_shard_iterator(0),16,{"return_inputs":False,"eval":True}) #
