from config_schema import HabitatConfig
from utils.habitat_worker import LoggingHabitatWorker
from dataclasses import asdict
from utils import measures
import numpy as np
USE_ORACLE=False
config = '/Projects/spatial_training/conf/habitat/objectnav_hm3d_v2_rgbd_semantic.yaml'
config = '/Projects/spatial_training/conf/habitat/objectnav_mp3d_rgbd.yaml'
config ="/Projects/spatial_training/conf/habitat/ovon_optim.yaml"
# config = '/Projects/configs/configs/objectnav_hssd-hab_rgbd.yaml'
# config = "/Projects/spatial_training/conf/habitat/vln_r2r.yaml"
config = '/Projects/spatial_training/conf/habitat/hm3d_training.yaml'
config = "/Projects/spatial_training/conf/habitat/objectnav_hm3d_rgbd_semantic.yaml"
cfg = HabitatConfig(
    config_path = config,
    workspace= '/Projects/spatial_training', # folder containing "data",
    split= 'train',
    auto_flush=False
    
)
# from ovon.dataset.objectnav_dataset import ObjectNavDatasetV2
worker = LoggingHabitatWorker(**asdict(cfg), logging_output_dir="",
    logger_actor=None,log_oracle=USE_ORACLE)

# from constants import episode_labels_table
# full_episodes = episode_labels_table['sample400']
# import json
# current_complete = json.load(open('oracle_durations_sample400.json','r'))
# remaining_episodes = [ep for ep in full_episodes if ep not in current_complete]
# print(f"Continuing from {len(current_complete)} completed episodes, {len(remaining_episodes)} remaining episodes.")
worker.assign_shard(None)
# worker.assign_shard()

# duration_dict = {}
# import time
# results_file = "hm3d_basic_speed.json"
# import json
# from tqdm import tqdm
# import random
# with open(results_file,'w') as f:
#     f.write('{')
#     for j in tqdm(range(400)):
#         t0 = time.time()
#         step_dict = worker.reset()
#         reset_time = time.time()-t0
#         t0 = time.time()

#         episode_label = step_dict['info']['episode_label']
#         i = 0
#         while not (step_dict['done'] or (step_dict['info']['oracle_action']<0 and USE_ORACLE)):
#             if USE_ORACLE:
#                 action = step_dict['info']['oracle_action']
#             else:
#                 action = random.choice([1,2,3])
#             step_dict = worker.step(action)
#             i+=1
#         # if step_dict['info']['success']:
#         duration = time.time()-t0
#         result = {
#             "steps": i,
#             "Tstep": duration,
#             "Treset": reset_time,
#             "freq": i/duration,
#             "worker_latency": duration/i,
#             "env_latency":np.array(worker.steps['env_latency']).mean()
#         }
        
#         f.write(f'"{episode_label}":{json.dumps(result)},\n')
#         f.flush()
#         duration_dict[episode_label] = i
       
#     f.write('}')