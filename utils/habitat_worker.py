from collections import defaultdict
import numpy as np
import regex as re
import os
import time
from contextlib import contextmanager
import json
import uuid
from PIL import Image
import time
import numpy as np
from typing import Optional, Any
from datasets import load_from_disk, load_dataset, Sequence
from datasets import Image as hugging_Image
from utils.bev_utils import load_depth, pos2grid, get_grid_pos

@contextmanager
def suppress_cpp_output():
    """
    Redirects C++ level stdout/stderr to /dev/null for the duration of the context.
    Critically, this works for libraries like Magnum/Habitat that write directly 
    to file descriptors 1 and 2, bypassing Python's sys.stdout.
    """
    # 1. Open /dev/null
    with open(os.devnull, "w") as devnull:
        # 2. Duplicate the original file descriptors (to restore later)
        # We save these so we can toggle logging back ON after init.
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        
        try:
            # 3. Overwrite stdout (1) and stderr (2) with the devnull FD
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            # 4. Restore the original FDs
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            # Close the duplicates
            os.close(old_stdout)
            os.close(old_stderr)
            
def apply_schema(data, schema):
    """
    Recursively filters 'data' to only include keys present in 'schema'.
    Structure matches schema; values in schema are ignored (placeholders).
    """
    if schema is None:
        return data
    result = {}
    for key, sub_schema in schema.items():
        if key in data:
            # If both are dictionaries, recurse to allow fine-grained filtering
            if isinstance(sub_schema, dict) and isinstance(data[key], dict):
                result[key] = apply_schema(data[key], sub_schema)
            else:
                # Leaf node: Return the data found at this key
                result[key] = data[key]
    return result


def save_run_video(steps_data, filename, output_dir, fps=4, quality=6,return_thumbnail=True):
    """
    Stateless helper function to render video.
    """
    from habitat.utils.visualizations import utils as vut

    if not steps_data or 'obs' not in steps_data:
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    action_list = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    
    # Process images
    # Note: We use the helper 'apply_schema' instead of self._apply_schema
    images = [
        vut.observations_to_image(apply_schema(obs, {"rgb": True}), info) 
        for obs, info in zip(steps_data['obs'], steps_data['info'])
    ]
    
    # Handle the appended action 0 logic from your original code
    # (Ensure we don't mutate the passed read-only object in a way that breaks things)
    actions = steps_data['action'] + [0]
    
    texts = [
        [
            f"episode: {steps_data['info'][0]['episode_label']} step: {idx}",
            action_list[int(action)],
            f"distance_to_goal: {info['distance_to_goal']}",
            f"distance_reward: {info['distance_to_goal_reward']}",
            f"goal: {obs['goal_name']}",
            f"spl: {steps_data['info'][-1]['spl']}",
        ]
        for idx, (action, obs, info) in enumerate(zip(actions, steps_data['obs'], steps_data['info']))
    ]
    
    images = [vut.overlay_text_to_image(image, text) for image, text in zip(images, texts)]

    vut.images_to_video(
        images=images,
        output_dir=output_dir,
        video_name=filename,
        fps=fps,
        quality=quality,
        verbose=False
    )
    if return_thumbnail:
        return os.path.join(output_dir, filename + ".mp4"),images[-1]

    return os.path.join(output_dir, filename + ".mp4")

summary_schema = {
    "episode_label":True, 
    "scene_id":True, #redundant, but convenient for scene based analysis
    "spl":True,
    "distance_to_goal":True,
    "soft_spl":True,
    "success":True,
}

default_logging_schema = {
    # 1. Observation: Keep only RGB and Goal Name (for text overlay)
    "obs": {
        "rgb": True,
        "goal_name": True,
        "patch_coords": True
    },
    
    # 2. Info: Keep metrics for text overlay and Map for visual overlay
    "info": {
        "distance_to_goal": True,
        "distance_to_goal_reward": True,
        "spl": True,
        "soft_spl":True,
        "episode_label": True,
        "episode_id": True,
        "scene_id": True,
        # Critical for drawing the map overlay in vut.observations_to_image
        "top_down_map": {
            "map": True,
            "fog_of_war_mask": True,
            "agent_map_coord": True,
            "agent_angle": True
        },
        "pos_rots":True,
        "success": True
    },
    
    # 3. Top-Level Primitives (Required for Metrics/Video)
    "action": True,    # Needed for text overlay
    "reward": True,    # Needed for 'mean_distance_reward' metric
    "done": True,      # Standard
    # 4. Extras (Injected by your Worker)
    "timestamp": True, # Needed for 'throughput' calculation
    "stuck": True,     # Needed for 'collision_rate'
    "fp_stop": True,   # Needed for guard metrics
    "fn_stop": True    # Needed for guard metrics
}

def supplementary_logging_helper(episode_data,result_row):
    # --- Supplemental Logs (User Defined Reductions) ---
    # Pattern: "sup/<reduction_type>/<name>s"
    # Supported: mean, max, min, median, sum, prod
    reduction_map = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "median": np.median,
        "sum": np.sum,
        "prod": np.prod
    }

    for key, values in episode_data.items():
        if key.startswith("sup/"):
            try:
                # Parse format: sup/mean/entropy -> type="mean", name="entropy"
                _, reduction_type, metric_name = key.split("/", 2)
                
                if reduction_type in reduction_map and len(values) > 0:
                    # Apply reduction to the list of values
                    # We cast to np.array to handle lists of floats/ints safely
                    val_array = np.array(values)
                    scalar_result = reduction_map[reduction_type](val_array)
                    
                    # Store as "sup/entropy" (stripping the reduction type for cleaner logging)
                    # or keep original key "sup/mean/entropy" if you prefer explicit naming.
                    # Here we keep the full key to avoid collision (e.g. mean/x vs max/x)
                    result_row[f"sup/{reduction_type}_{metric_name}"] = scalar_result
            except ValueError:
                # Malformed key (e.g. "sup/broken_format"), skip it
                continue
            except Exception as e:
                # Value error (e.g. non-numeric data in reduction), skip it
                print(f"Failed to reduce user key {key}: {e}")
                continue

def habitat_logging_helper(episode_data):
    '''
    Docstring for habitat_logging_helper
    
    :param episode_data: Description

    returns: 
    - episode_logs: dictionary of scalar data for the entire episode
    - sequence_logs dictionary of lists/arrays for granular sequence data
    '''
    episode_logs = apply_schema(episode_data['info'][-1],summary_schema) 
    #TODO: check safety. does habitat change the internal episode the moment step(stop) is called? if so, we are in trouble
    # safety overrides for now, using first step guarantees correctness
    episode_logs['episode_label'] = episode_data['info'][0]['episode_label']
    episode_logs['episode_id'] = episode_data['info'][0]['episode_id']
    episode_logs['scene_id'] = episode_data['info'][0]['scene_id']
    episode_logs['goal'] = episode_data['obs'][0]['goal_name'] #need this for obvious reasons

    episode_logs['n_steps'] = len(episode_data['action'])
    episode_logs['duration'] = episode_data['timestamp'][-1]-episode_data['timestamp'][0]
    episode_logs['throughput'] = episode_logs['n_steps']/max(episode_logs['duration'],1e-5)

    episode_logs['fpg_trigger_count'] = np.sum(np.array(episode_data['fp_stop'])) #works thanks to defaultdict shenanigans
    episode_logs['fng_trigger_count'] = np.sum(np.array(episode_data['fn_stop'])) #works thanks to defaultdict shenanigans

    # sequence level logs
    raw_collisions = episode_data.get('stuck',[]) #this is already aligned to actions, no need to slice. 
    # auxiliary performance metrics
    episode_logs['collision_rate'] = np.sum(np.array(raw_collisions))/episode_logs['n_steps']
    episode_logs['mean_distance_reward'] = np.mean(np.array(episode_data['reward'][1:-1])) # slice to exclude the terminal reward
    # save the video.
    supplementary_logging_helper(episode_data,episode_logs)

    sequence_logs = {}
    sequence_logs['action_history'] = episode_data['action']
    sequence_logs['positions'] = [info['pos_rots'][:3] for info in episode_data['info'][:-1]] #final position is redundant due to stop action
    sequence_logs['quaterions'] = [info['pos_rots'][3:] for info in episode_data['info'][:-1]] #doubt this is meaningful in wandb but it can't cost that much storage right?
    sequence_logs['collision_events'] = raw_collisions

    return episode_logs, sequence_logs


    # vid_path,img = save_run_video(episode_data,result_row['episode_label'],output_dir,return_thumbnail=True, **video_kwargs) 
    # result_row['vid/episode_video'] = vid_path
    # result_row['img/thumbnail'] = Image.fromarray(img)

# # from collections import deque
# # --- Helper Function ---
# def get_scene_id(scene_path):
#     scene_id = scene_path.split("/")[-1]
#     scene_id = re.sub(r'\.basis\.glb$', '', scene_id)
#     return scene_id
# robust version
def get_scene_id(scene_path):
    # os.path.basename handles both "/" and "\" correctly
    filename = os.path.basename(scene_path)
    # Optional: Make regex handle both .basis.glb and standard .glb
    scene_id = re.sub(r'(\.basis)?\.glb$', '', filename)
    return scene_id

class HabitatWorker:
    def __init__(self, assigned_episode_labels=None,workspace='/Projects/SG_VLN_HumanData/SG-VLN', config_path="configs/objectnav_hm3d_rgbd_semantic.yaml", enable_caching=True,dataset_path = None, scenes_dir=None,split="val", debug_with_humanaction=False, postprocess= True,output_schema=None,logging_schema=None,fn_guard=False,fp_guard=False,voxel_kwargs=None):
        from habitat.config.default import get_config
        from habitat.config import read_write
        from habitat.config.default_structured_configs import (
            TopDownMapMeasurementConfig,
            FogOfWarConfig,
        )
        from habitat import Env, make_dataset
        """
        assigned_episode_labels: List of strings ['scene_id_episode_id', ...] specific to this worker.
        enable_caching: If True, stores observations for video generation.
        """
        if workspace is not None:
            # nuclear option to ensure proper habitat loading
            import os
            os.chdir(workspace)

        self.postprocess = postprocess

        self.enable_caching = enable_caching
        if enable_caching:
            self.steps = defaultdict(list)
            if logging_schema is None:
                self.logging_schema = default_logging_schema
            else:
                self.logging_schema = logging_schema

        self.output_schema = output_schema
        self.logging_actor = None

        self.fn_guard = fn_guard
        self.fp_guard = fp_guard
        self.debug_with_humanaction = debug_with_humanaction
        if self.debug_with_humanaction:
            # eval_dataset_dir = '/Projects/SG_VLN_HumanData/SG-VLN/data/sft_datasets/train'
            eval_dataset_dir = '/Projects/SG_VLN_HumanData/SG-VLN/data/sft_datasets/17DRP5sb8fy_json'
            if not os.path.exists(os.path.expanduser(eval_dataset_dir)) and "/" in eval_dataset_dir:
                human_dataset = load_dataset(eval_dataset_dir, split="val")
            else:
                human_dataset = load_from_disk(eval_dataset_dir)
            self.human_dataset = human_dataset.cast_column("images", Sequence(hugging_Image(decode=True)))

        self.voxel_kwargs = voxel_kwargs
        # --- Initialize Config & Env ---
        self.config_env = get_config(config_path)
        with read_write(self.config_env):
        # --- OVERRIDE PATHS IF PROVIDED ---
            if dataset_path:
                # 1. The Episode Dataset (The JSON file)
                # Example: "/mnt/data/habitat/datasets/objectnav/val.json.gz"
                self.config_env.habitat.dataset.data_path = dataset_path
            else:
                self.config_env.habitat.dataset.split = split

            if scenes_dir:
                # 2. The Scene Assets (The folder containing .glb files)
                # Example: "/mnt/data/habitat/scene_datasets/"
                self.config_env.habitat.dataset.scenes_dir = scenes_dir
            # Add TopDownMap for visualization
            self.config_env.habitat.task.measurements.top_down_map = (
                TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=512,
                    draw_goal_positions=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_border=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5, fov=72),
                )
            )

        # Create Dataset
        self.full_dataset = make_dataset(
            self.config_env.habitat.dataset.type, config=self.config_env.habitat.dataset
        )
        if assigned_episode_labels is not None:
            self.assign_shard(assigned_episode_labels)
        else:
            print("skipping sim initialization since no shards provided, please call assign_shard")

    def assign_shard(self,assigned_episode_labels = None):
        from habitat.core.dataset import EpisodeIterator
        from habitat.gym import make_gym_from_config
        
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        self.steps = defaultdict(list)
        self.last_step = None
        self.assigned_labels = set(assigned_episode_labels) if assigned_episode_labels is not None else None

        if self.assigned_labels is not None:
            # Filter Episodes
            def filter_fn(eps):
                scene_id = get_scene_id(eps.scene_id)
                episode_label = f'{scene_id}_{eps.episode_id}'
                return episode_label in self.assigned_labels

            dataset = self.full_dataset.filter_episodes(filter_fn)
        else:
            dataset = self.full_dataset
        try:
            self.shard_length = dataset.num_episodes
        except:
            self.shard_length = len(dataset)

        if self.shard_length == 0:
            print("WARNING: Assigned shard is empty!")
        self.episode_counter = 0 #distinguish from "steps" concept per episode

        # Initialize Env
        # self.env = Env(self.config_env, dataset)
        with suppress_cpp_output():
            self.env = make_gym_from_config(self.config_env,dataset)
        # self.env = make_gym_from_config(self.config_env,dataset)

        # Setup Iterator 
        self.env.habitat_env.episode_iterator = EpisodeIterator(
            dataset.episodes,
            cycle=True,
            shuffle=True,
            group_by_scene=False,
            seed=17,
            # max_scene_repeat_episodes=10
        )
        print(f"Actor assigned with shard of {len(dataset.episodes)} episodes.")

    def init_human_data(self):
        episode_id = self.env.habitat_env._current_episode.raw_episode_id
        human_data = self.human_dataset.filter(lambda x: x['episode_id'] == episode_id)
        cur_human_data = human_data['validation'][0]
        self.cur_human_data = cur_human_data
        return 
    
    def get_human_data(self):
        return self.cur_human_data
    
    def get_human_depth(self, step_count):
        depth = load_depth(self.cur_human_data['depth_sequence'][step_count])
        return depth

    def get_human_rgb(self, step_count):
        rgb = self.cur_human_data['images'][step_count]
        return rgb

    def get_human_posrot(self, step_count):
        pos_rots = self.cur_human_data['pos_rots'][step_count]
        return pos_rots
    
    def get_human_action(self, step_count):
        agent_gt_traj = self.env.habitat_env._current_episode.reference_replay
        action_keys = {'STOP':0, 'MOVE_FORWARD':1, 'TURN_LEFT':2, 'TURN_RIGHT':3, 'LOOK_UP':4, 'LOOK_DOWN':5}
        human_action = action_keys[agent_gt_traj[step_count+1]['action']]
        return human_action
    
    def get_step_data(self):
        return self.steps

    def step_by_pose(self, position, rotation):
        '''
        In dataset, we save rotation as [x,y,z,w]
        Habitat need: [w,x,y,z]
        '''
        import habitat
        import quaternion
        # 1. Access the underlying simulator
        sim = self.env.unwrapped.habitat_env.sim
        
        # 2. Teleport the agent
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = [position[0], position[1]-0.88, position[2]]
        state.rotation = quaternion.from_float_array([rotation[3],rotation[0],rotation[1],rotation[2]])
        agent.set_state(state)
        
        # 3. Trigger the Task Update
        # We pass a 'None' or 'Empty' action to avoid moving again, 
        # but this forces Habitat to recalculate rewards, metrics, and sensors.
        # Note: 'action' usually needs to be in the format the env expects (e.g., 0 or a dict)
        # dummy_action = 0 
        # obs, reward, done, info = self.env.step(dummy_action)
        obs = sim.get_sensor_observations()
        measurements = self.env.unwrapped.habitat_env.task.measurements.get_metrics()
        reward = 0.0
        done = False
        info = measurements
        
        # state = self.env.habitat_env.sim.get_agent_state()
        # cam_state = self.env.habitat_env.sim.get_agent(0)._sensors['rgb'].node
        # T_np = np.array(cam_state.absolute_transformation())
        # pos = T_np[:3, 3]
        # rot = state.rotation  # Quaternion

        # print('before set: ', position, rotation)
        # print('after set: ', pos, [rot.x, rot.y, rot.z, rot.w])
        return obs, reward, done, info
        
    def step(self, action:int, step_count:int, supplementary_logs={}):
        """
        Standard step. 
        Args:
            action: The action id to take.
            output_schema: sets the output schema
            fp_guard: guard against false positive "stop“ calls with oracle
            fn_guard: guard against false negative "stop" calls with oracle
            supplementary logs: dict of extra info to log for this step. useful for recording action logprobs, agent inference latency, etc
                - WARNING: if you provide this, you MUST provide it every step, with the same keys. otherwise your data won't align properly!
        """
        last_distance = self.last_step['info']['distance_to_goal']
        extras = {'fp_stop':-1*int(not self.fp_guard),'fn_stop':-1*int(not self.fn_guard)} # -1 for not enabled, 0 for enabled but not triggerd.
        # oracle stop guards useful for reducin eval noise. #TODO: use habitat config instead of magic number
        if self.fp_guard and action==0 and last_distance > 0.1:
            action = np.random.choice([1,2,3]) #chose random non stop action
            extras['fp_stop'] = 1 #record the false positive incident
        if last_distance<0.1 and action!=0:
            if self.fn_guard: 
                action = 0
            extras['fn_stop'] = 1
        if self.debug_with_humanaction:
            action = self.get_human_action(step_count)
        import time
        # t0 = time.time()    
        # print(f"stepping env took {time.time()-t0}")   
        # t0 = time.time()

        if self.debug_with_humanaction and step_count < len(self.cur_human_data['depth_sequence'])-1:
            human_pos_rots = self.get_human_posrot(step_count+1)
            obs, reward, done, info = self.step_by_pose(human_pos_rots[:3], human_pos_rots[3:])
            obs['rgb'] = self.get_human_rgb(step_count+1)
            obs['depth'] = self.get_human_depth(step_count+1)
            obs['pos_rots'] = human_pos_rots
        else:
            obs, reward, done, info = self.env.step(action)
            obs['depth'] *= 5.0

        step_dict = {
        "obs":obs,
        "reward": reward,
        "done": done,
        "info": info
        }

        # print('step: ', step_count, action)
        if self.postprocess:
            step_dict = self._postprocess_step(step_dict)
            agent_pos, agent_heading = get_grid_pos(np.array(step_dict['info']['pos_rots']).reshape(1,-1), self.agent_start_loc)
            step_dict['obs']['agent_loc'] = agent_pos[0]
            step_dict['obs']['agent_heading'] = agent_heading[0]
            step_dict['obs']['step'] = step_count + 1
            extras['stuck'] = np.linalg.norm(np.array(self.last_step['info']['pos_rots'])-np.array(step_dict['info']['pos_rots']))<1e-5 # record collisions

        if self.enable_caching:
            extras["timestamp"] = time.time()
            if supplementary_logs is None:
                    supplementary_logs = {}
            else:
                # Automatically namespace user logs to prevent collisions
                # Example: User passes {"mean/entropy": 0.5} -> converts to "sup/mean/entropy"
                supplementary_logs = {
                    (f"sup/{k}" if not k.startswith("sup/") else k): v 
                    for k, v in supplementary_logs.items()
                }
            
            self._cache_step(self._apply_schema(step_dict,self.logging_schema) | supplementary_logs | extras)
            self.steps['action'].append(action)
        self.last_step = step_dict
        # print(f"postprocessing took {time.time()-t0}")
        return self._apply_schema(step_dict,self.output_schema)

    def get_last_step(self,output_schema=None):
        """Returns the current observation without stepping the environment."""
        return self._apply_schema(self.last_step,output_schema)
    
    def get_episodes(self):
        return self.env.episodes
    
    def reset(self, episode_id=None,output_schema=None,logging_schema=None):
        """
        Args:
            episode_id: please don't try this.
            output_schema: use and set the step output schema
        """
        # Access the habitat core environment
        self.steps = defaultdict(list)  # Clear video cache for new episode

        if episode_id is not None:
            # Find specific episode
            episodes = self.env.habitat_env._dataset.episodes
            episode = next((e for e in episodes if e.episode_id == str(episode_id)), None)
            self.env.habitat_env._current_episode = episode

        if output_schema is not None:
            self.output_schema = output_schema
        if logging_schema is not None:
            self.logging_schema = logging_schema
        # Force the iterator to point here
        # Note: Habitat iterators are complex; the safest way is to force the current episode
        with suppress_cpp_output():
            obs,info = self.env.reset(return_info=True)
        # obs,info = self.env.reset(return_info=True)
        self.episode_counter+=1
        # Calling reset() now loads 'current_episode'
        step_dict = {
        "obs":obs,
        "reward": 0,
        "done": False,
        "info": info
        }
        if self.debug_with_humanaction:
            self.init_human_data()
            step_dict['obs']['rgb'] = self.get_human_rgb(0)
            step_dict['obs']['depth'] = self.get_human_depth(0)
            step_dict['obs']['pos_rots'] = self.get_human_posrot(0)
        else:
            step_dict['obs']['depth'] *= 5.0
        if self.postprocess:
            step_dict = self._postprocess_step(step_dict)
            self.agent_start_loc = pos2grid(np.array(step_dict['info']['pos_rots']).reshape(1,-1))
            agent_pos, agent_heading = get_grid_pos(np.array(step_dict['info']['pos_rots']).reshape(1,-1), self.agent_start_loc)
            step_dict['obs']['agent_loc'] = agent_pos[0]
            step_dict['obs']['agent_heading'] = agent_heading[0]
            step_dict['obs']['step'] = 0
        if self.enable_caching:
            self._cache_step(self._apply_schema(step_dict,self.logging_schema) | {"timestamp": time.time()})
        self.last_step = step_dict
        return self._apply_schema(step_dict,self.output_schema)

    def save_video(self, output_dir, fps=4,quality = 3):
        filename = self.steps['info'][0]['episode_label']
        video_path,thumbnail = save_run_video(self.steps,filename,output_dir,fps,quality)
        # self.steps = defaultdict(list) # clear buffer after save
        return video_path, thumbnail

    def _cache_step(self, step):
        """Internal helper to format observations for make_video."""
        # Habitat's make_video expects a specific structure or just a list of dicts/arrays
        # We store the raw obs; make_video handles extracting 'rgb'
        for key, value in step.items():
            self.steps[key].append(value)

    def getattr(self, attr_name):
        """Proxy to access attributes of the underlying Env."""
        return getattr(self.env, attr_name)
    
    def get_metrics(self):
        """Get episode metrics"""
        if hasattr(self.env, 'get_metrics'):
            return self.env.get_metrics()
        return {}

    def _get_goal_name(self, goal_index):
        """
        Decodes the integer objectgoal from the observation into a string.
        """
        # In Habitat ObjectNav, the goal index maps to a list of categories 
        # defined in the dataset config/mapping.
        # We can access the mapping via the task's dataset.
        
        # Access the underlying dataset attribute mapping
        # Note: 'category_to_task_category_id' maps Name -> ID. We need the reverse.
        if not hasattr(self, '_id_to_category_name'):
            # Build the cache once
            dataset = self.env.habitat_env._dataset
            mapping = dataset.category_to_task_category_id
            self._id_to_category_name = {v: k for k, v in mapping.items()}
        
        # Handle scalar or single-element array input
        if hasattr(goal_index, 'item'):
            goal_index = goal_index.item()
            
        return self._id_to_category_name.get(goal_index, f"Unknown({goal_index})")

    def is_exhausted(self):
        return self.episode_counter >= self.shard_length

    def _postprocess_step(self,step_dict):
        obs = step_dict['obs']
        info = step_dict['info']
        if 'objectgoal' in obs:
            obs['goal_name'] = self._get_goal_name(obs['objectgoal'])
        
        if self.env.habitat_env.sim:
            state = self.env.habitat_env.sim.get_agent_state()

            # pos = state.position  # Vector3
            cam_state = self.env.habitat_env.sim.get_agent(0)._sensors['rgb'].node
            T_np = np.array(cam_state.absolute_transformation())
            pos = T_np[:3, 3]
            rot = state.rotation  # Quaternion

            info['pos_rots'] = [
                float(pos[0]), float(pos[1]), float(pos[2]),      # x, y, z
                float(rot.x), float(rot.y), float(rot.z), # qx, qy, qz
                float(rot.w)                                 # w
            ]
            # state = self.env.habitat_env.sim.get_agent(0)._sensors['rgb'].node
            # T_np = np.array(state.absolute_transformation())
            # pos = T_np[:3, 3]
            # rot_matrix = T_np[:3, :3]
            # rot = R.from_matrix(rot_matrix).as_quat()
            # info['pos_rots'] = [
            #     float(pos[0]), float(pos[1]), float(pos[2]),      # x, y, z
            #     float(rot[0]), float(rot[1]), float(rot[2]), # qx, qy, qz
            #     float(rot[3])                                 # w
            # ]

            # if 'pos_rots' in obs:
            #     print('info: ', info['pos_rots'], 'obs: ', obs['pos_rots'])
            #     print('diff: ', [info['pos_rots'][i] - obs['pos_rots'][i] for i in range(len(info['pos_rots']))])
                # assert info['pos_rots'] == obs['pos_rots']
            if self.voxel_kwargs is not None:
                from utils.bev_utils import get_patch_coords
                H,W = step_dict['obs']['depth'].shape[:2] 
                pos_rots = np.array([obs['pos_rots']]) if 'pos_rots' in obs else np.array([info['pos_rots']])
                obs['patch_coords'] = get_patch_coords(pos_rots, step_dict['obs']['depth'].reshape((1,H,W)),**self.voxel_kwargs)[0] # 1 by H by W by 3 patch coords
        info['scene_id']=get_scene_id(self.env.current_episode().scene_id)
        info['episode_id']=self.env.current_episode().episode_id
        info['episode_label']=f"{info['scene_id']}_{info['episode_id']}"
        curr_idx = self.episode_counter-1
        info['epoch']= curr_idx // self.shard_length
        info['step_in_epoch']=curr_idx % self.shard_length
        # 2. Provide Semantic Mapping (ID -> Label), too expensive, should just request once instead of always send.
        # The driver can now map any pixel in obs['semantic'] to a string.
        # if 'semantic' in obs:
        #     info['semantic_mapping'] = self.get_semantic_label_mapping()
        return self._sanitize(step_dict)
    
    def get_semantic_label_mapping(self):
        """
        Returns a dictionary mapping the Integer ID found in the semantic sensor 
        to the String Category Name.
        
        Format: {104: "chair", 105: "chair", 106: "table", ...}
        """
        # Return cached mapping if the scene hasn't changed
        current_scene = self.env.current_episode().scene_id
        if getattr(self, '_cached_scene_id', None) == current_scene and \
        hasattr(self, '_cached_semantic_mapping'):
            return self._cached_semantic_mapping

        # Double check sim exists
        if not self.env.habitat_env.sim:
            return {}

        # Access the Scene Graph
        semantic_scene = self.env.habitat_env.sim.semantic_scene
        objects = semantic_scene.objects
        
        # Build the mapping: Instance ID (Index) -> Category Name
        # Note: Habitat Instance IDs correspond to the index in the objects list.
        mapping = {}
        for i, obj in enumerate(objects):
            if obj and obj.category:
                mapping[i] = obj.category.name()
            else:
                mapping[i] = "unknown"

        # Cache it to avoid rebuilding this large dict every step
        self._cached_scene_id = current_scene
        self._cached_semantic_mapping = mapping
        
        return mapping

    def _apply_schema(self,data,schema):
        return apply_schema(data,schema)
    
    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()
            self.env = None

    def _sanitize(self, data):
        """
        Recursively converts Habitat/Magnum types into standard Python/Numpy types
        to ensure the Driver (which lacks Habitat) can unpickle the result.
        """
        # 1. Handle Dictionaries (and things that act like them, e.g. Habitat Observations)
        if isinstance(data, dict) or hasattr(data, "keys"):
            # Force conversion to standard dict to strip custom class types
            return {k: self._sanitize(v) for k, v in dict(data).items()}
        
        # 2. Handle Lists/Tuples
        elif isinstance(data, (list, tuple)):
            return [self._sanitize(v) for v in data]
        
        # 3. Handle Numpy (Safe to pass as-is, assuming Driver has Numpy)
        elif isinstance(data, (np.ndarray, np.generic)):
            return data
        
        # 4. Handle Magnum Vectors/Quaternions (Common in Habitat Info)
        # We detect them by checking if the module name starts with 'magnum'
        elif hasattr(type(data), "__module__") and type(data).__module__.startswith("magnum"):
            # Magnum types are usually iterable (Vector3 -> [x,y,z])
            return np.array(list(data))
        # 5. Handle Basic Primitives (int, float, str, bool, None)
        return data

# Inherit from your existing base worker
# from your_project import HabitatWorker, default_habitat_log_task

class LoggingHabitatWorker(HabitatWorker):
    
    def __init__(
        self, 
        *args, 
        logging_output_dir: str,
        logger_actor: Any = None,
        **kwargs
    ):
        from pathlib import Path
        self.log_dir = Path(logging_output_dir).resolve()

        # Use the new default_logging_schema provided in your prompt if none is passed
            
        super().__init__(*args, **kwargs)
        
        self.logger_actor = logger_actor
        os.makedirs(self.log_dir, exist_ok=True)

    def _flush_logs_to_disk(self):
        """
        Orchestrates saving heavy artifacts to disk and sending lightweight references to Ray.
        """
        if len(self.steps['action'])==0:
            print("log flush called, but step cache is empty! skipping...")
            return
        # 1. Resolve Naming & Directory
        # Use first step info for stable IDs (scene, episode)
        first_info = self.steps['info'][0] if self.steps['info'] else {}
        scene_id = first_info.get('scene_id', 'unknown_scene')
        ep_label = first_info.get('episode_label', f"ep_{int(time.time())}")
        
        # Create dedicated folder: /logs/scene_X/episode_label/
        # This keeps artifacts (video, thumb, json) grouped together
        save_dir = os.path.join(self.log_dir, scene_id, str(ep_label))
        os.makedirs(save_dir, exist_ok=True)

        # 2. Calculate Metrics & Split Data
        # Returns scalars (episode_logs) and raw lists (sequence_logs)
        episode_logs, sequence_logs = habitat_logging_helper(self.steps)

        # 3. Generate & Save Video + Thumbnail
        # Uses your helper. Returns (video_path_string, PIL_Image)
        vid_path, thumb_img = save_run_video(
            steps_data=self.steps,
            filename="video", # save_run_video adds extension
            output_dir=save_dir,
            quality=4,
            return_thumbnail=True
        )
        
        # Manually save the thumbnail object to disk
        thumb_path = os.path.join(save_dir, "thumbnail.jpg")
        Image.fromarray(thumb_img).save(thumb_path, quality=85)

        # 4. Save Raw Sequence Data (The "Raw Data" Fix)
        # Dump trajectory/actions to JSON so we don't lose granular history
        seq_path = os.path.join(save_dir, "trace.json")
        
        # Convert numpy types to native python for JSON serialization
        def default_serializer(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)

        with open(seq_path, 'w') as f:
            json.dump(sequence_logs, f, default=default_serializer)

        with open(os.path.join(self.log_dir,"results"),'a') as f:
            f.write(json.dumps(episode_logs,default=default_serializer)+"\n") #save the result

        # 5. Construct Global Payload (Paths + Scalars)
        # Merge the scalar metrics with the file paths
        payload = {
            **episode_logs,
            "vid/episode_video": vid_path,  # Path for WandB Video
            "img/thumbnail": thumb_path,    # Path for WandB Image
            "raw/trace": seq_path,          # Path for WandB Artifact/File
            
            # System Metadata
            "worker_pid": os.getpid(),
            "node": os.uname().nodename,
            "timestamp": time.time()
        }

        # 6. Send to Global Logger
        if self.logger_actor:
            self.logger_actor.log_row.remote(row=payload)

    def reset(self, episode_id=None,output_schema=None,logging_schema=None):
        if len(self.steps['action'])>0:
            self._flush_logs_to_disk()
        return super().reset(episode_id,output_schema,logging_schema)

    def assign_shard(self, assigned_episode_labels=None):
        if len(self.steps['action'])>0:
            self._flush_logs_to_disk()
        return super().assign_shard(assigned_episode_labels)
    
if __name__ == "__main__":
    import time
    import numpy as np
    import os


    # --- Configuration ---
    LOG_OUTPUT_DIR = "/Projects/SG_VLN_HumanData/spatial_training/dump/test"
    MAX_TEST_STEPS = 100 # Safety cap
    

    print(f"\n=== Starting LoggingHabitatWorker Test ===")

    # 1. Mock the Ray Logging Actor
    # This simulates the remote receiving end of your pipeline
    class MockLoggingActor:
        class RemoteHandle:
            def remote(self, row):
                print(f"\n[MockLogger] >> SIGNAL RECEIVED! Worker sent payload:")
                for k, v in row.items():
                    # Check if it's a path or a scalar
                    val_type = "SCALAR"
                    if isinstance(v, str) and (os.path.sep in v or "." in v):
                        val_type = "PATH  "
                    print(f"  [{val_type}] {k:<25}: {v}")

        def __init__(self):
            self.log_row = self.RemoteHandle()

    mock_actor = MockLoggingActor()

    # 2. Initialize the New Worker
    # We pass the default schema explicitly to ensure the worker captures 
    # exactly what we expect (RGB + Metrics + Map)
    print("Initializing LoggingHabitatWorker...")
    worker = LoggingHabitatWorker(
        workspace='/Projects/SG_VLN_HumanData/SG-VLN', 
        enable_caching=True, 
        postprocess=True,
        logging_output_dir=LOG_OUTPUT_DIR,
        logger_actor=mock_actor,
        logging_schema=default_logging_schema # Defined in your script above
    )
    
    print("Resetting Environment...")
    first_obs = worker.reset()

    # 3. Payload Check (Sanity Check)
    # Ensure the worker is actually producing RGB data to log
    if 'obs' in worker.steps and len(worker.steps['obs']) > 0:
        rgb_sample = worker.steps['obs'][0].get('rgb')
        if rgb_sample is not None:
            print(f"Worker is buffering RGB: {rgb_sample.shape} | {rgb_sample.nbytes / 1024**2:.2f} MB")
        else:
            print("WARNING: RGB not found in worker buffer!")

    # 4. Run Execution Loop (Until Done)
    print(f"\nRunning Episode (Max {MAX_TEST_STEPS} steps)...")
    
    t_start = time.time()
    steps_completed = 0
    done = False
    
    while not done and steps_completed < MAX_TEST_STEPS:
        # Action 1 = Move Forward (usually)
        # We mix in some turns (2, 3) to make the video interesting
        action = np.random.choice([1, 2, 3], p=[0.8, 0.1, 0.1])
        
        # === THE CRITICAL PART ===
        # The worker.step() method contains the trigger logic.
        # When 'done' becomes True, it will auto-flush to disk.
        step = worker.step(action)
        reward = step['reward']
        done = step['done']
        steps_completed += 1
        
        if steps_completed % 10 == 0:
            print(f"Step {steps_completed}: Action {action}, Reward {reward:.4f}...", end='\r')

    t_end = time.time()
    duration = t_end - t_start
    
    print(f"\n\n=== Episode Finished ===")
    print(f"Status: {'DONE (Natural)' if done else 'STOPPED (Max Steps)'}")
    print(f"Steps: {steps_completed}")
    print(f"Duration: {duration:.4f}s (FPS: {steps_completed/duration:.2f})")

    # 5. Verification
    # If the episode finished naturally, the MockLogger should have printed above.
    # We also verify the files exist on disk.
    worker._flush_logs_to_disk() # Uncomment to force test if needed

    print("\n=== Verifying Artifacts on Disk ===")
    
    # Find the generated folder (it uses scene_id/episode_label structure)
    # We just walk the log dir to find it
    found_video = False
    found_trace = False
    
    for root, dirs, files in os.walk(LOG_OUTPUT_DIR):
        for file in files:
            path = os.path.join(root, file)
            size_mb = os.path.getsize(path) / 1024 / 1024
            
            if file.endswith(".mp4"):
                print(f"  [VIDEO] Found: {file} ({size_mb:.2f} MB)")
                found_video = True
            elif file.endswith(".json"):
                print(f"  [TRACE] Found: {file} ({size_mb:.2f} MB)")
                found_trace = True
            elif file.endswith(".jpg"):
                print(f"  [THUMB] Found: {file} ({size_mb:.2f} MB)")

    if not done:
        print("\nNOTE: Episode did not finish naturally, so flush may not have triggered.")
        print("To test flush manually, you can call: worker._flush_logs_to_disk()")

    print("\nTest Complete.")
    worker.close()

