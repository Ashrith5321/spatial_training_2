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
    Filters data based on schema.
    
    Special Logic:
    - Keys starting with '+' (e.g. '+my_metric') are treated as "Default Allow".
      1. The '+' is stripped from the output key.
      2. The key is KEPT unless the schema explicitly sets the stripped name to False.
    """
    if schema is None:
        for key in data.keys():
            if key.startswith('+'):
                data[key[1:]]=data.pop(key)
        return data
        
    result = {}
    
    for key, value in data.items():
        # --- 1. Handle "Default Allow" Keys (Prefix '+') ---
        if isinstance(key, str) and key.startswith('+'):
            clean_key = key[1:] # Strip '+'
            
            # Check if schema explicitly BANS this key
            # If clean_key is NOT in schema, we keep it (Default True)
            # If clean_key IS in schema, we obey the schema (True/False)
            if schema.get(clean_key, True) is not False:
                result[clean_key] = value
            continue

        # --- 2. Handle Standard Keys ---
        if key in schema:
            sub_schema = schema[key]
            
            # Recurse for nested dicts
            if isinstance(sub_schema, dict) and isinstance(value, dict):
                result[key] = apply_schema(value, sub_schema)
            
            # Keep if True
            elif sub_schema:
                result[key] = value
                
    return result

from enum import Enum, auto

class TaskType(Enum):
    OBJECTNAV = auto()  # RGB + Object Category ID
    VLN = auto()        # RGB + Instruction Text (R2R/RxR)
    IMAGENAV = auto()   # RGB + Goal Image
    GENERIC = auto()    # PointNav or unknown

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
    try:
        action_probs = steps_data["sup/action_probs"] + [[0,0,0,0]]
    except:
        action_probs = [[0,0,0,0]]*len(actions)

    texts = [
        [
            f"episode: {steps_data['info'][0]['episode_label']} step: {idx}",
            f"action: {action_list[int(action)]} @ P={probs[int(action)]:.3f}",
            f"probs: {[f'{prob:.3f}' for prob in probs]}",
            f"oracle: {action_list[int(info.get('oracle_action',0))]}",
            f"distance_to_goal: {info.get('distance_to_goal',None)}",
            f"distance_reward: {info.get('distance_to_goal_reward',None)}",
            f"goal: {obs['instr_or_goal']}",
            f"spl: {steps_data['info'][-1].get('spl',None)}",
        ]
        for idx, (action,probs,obs, info) in enumerate(zip(actions, action_probs, steps_data['obs'], steps_data['info']))
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

# summary_schema = {
#     "episode_label":True, 
#     "scene_id":True, #redundant, but convenient for scene based analysis
#     "spl":True,
#     "distance_to_goal":True,
#     "soft_spl":True,
#     "success":True,
# }

default_logging_schema = {
    # 1. Observation: Keep only RGB and Goal Name (for text overlay)
    "obs": {
        "rgb": True,
        "instr_or_goal": True,
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

def habitat_logging_helper(episode_data,summary_schema):
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
    episode_logs['goal'] = episode_data['obs'][0]['instr_or_goal'] #need this for obvious reasons

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
    sequence_logs['distance_to_goal'] = [info['distance_to_goal'] for info in episode_data['info']]
    copy_keys = ['reward','fp_stop','fn_stop']
    for k in copy_keys:
        sequence_logs[k] = episode_data[k]

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

class ObjectNavOracle:
    def __init__(self, env, success_distance,goal_radius=0.2,goal_measure="distance_to_goal"):
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        """
        High-performance Oracle using MultiGoalShortestPath and episode caching.
        
        :param env: Habitat Environment. MUST NOT be RL wrapped.
        :param success_distance: Distance threshold to consider goal reached
        :param goal_radius: Radius for the low-level ShortestPathFollower
        """
        self.sim = env.sim
        self.env = env
        self.success_distance = success_distance
        self.goal_measure = goal_measure

        # We assume goal_radius for the low-level follower is similar to success_dist
        # You can tune this if the agent stops too early/late.
        self.follower = ShortestPathFollower(
            self.sim, 
            goal_radius=goal_radius, 
            return_one_hot=False
        )
        
        # Caching mechanisms
        self._cached_episode_id = None
        self._cached_candidates = None # List of Vector3 targets

    def _refresh_candidates(self, episode):
        import numpy as np
        """
        Parses episode goals once and stores all valid navigable target points.
        """
        candidates = []
        for goal in episode.goals:
            # 1. Prefer explicit ViewPoints (guaranteed navigable locations)
            if hasattr(goal, 'view_points') and goal.view_points:
                candidates.extend([self.sim.pathfinder.snap_point(vp.agent_state.position) for vp in goal.view_points])
            else:
                # 2. Fallback: Snap object position to NavMesh
                snapped_pos = self.sim.pathfinder.snap_point(goal.position)
                # Check if snap was successful (isnan check is robust for Vector3)
                if not np.isnan(snapped_pos[0]):
                    candidates.append(snapped_pos)
        
        # Fallback if metadata is totally broken (prevents crash)
        if not candidates:
            candidates = [g.position for g in episode.goals]
            
        self._cached_candidates = candidates
        self._cached_episode_id = episode.episode_id

    def get_best_action(self):
        import habitat_sim
        """
        Returns the optimal action index (int).
        Runtime: < 2ms (vs ~300ms unoptimized).
        """
        episode = self.env.current_episode
        agent_state = self.sim.get_agent_state()
        # 1. Cache Check
        if self._cached_episode_id != episode.episode_id:
            self._refresh_candidates(episode)
            
        # If no candidates exist (broken episode), Stop.
        if not self._cached_candidates:
            return 0 

        agent_pos = agent_state.position

        # 2. MultiGoal Shortest Path (C++ backend)
        # Finds the closest point among all candidates in one pass.
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = agent_pos
        path.requested_ends = self._cached_candidates
        
        found = self.sim.pathfinder.find_path(path)

        # 3. Decision Logic
        if not found:
            # Agent is on an island or navmesh is broken. Return STOP (0) or Forward(1)
            return 0 
        distance_to_goal = self.env.task.measurements.measures[self.goal_measure].get_metric()
        if distance_to_goal is not None and distance_to_goal < self.success_distance:
            return 0  # STOP
        best_target = path.points[-1]
        
        return self.follower.get_next_action(best_target)
    
class HabitatWorker:
    def __init__(self, assigned_episode_labels=None,workspace='/Projects/SG_VLN_HumanData/SG-VLN', config_path="configs/objectnav_hm3d_rgbd_semantic.yaml", enable_caching=True,dataset_path = None, scenes_dir=None,split="val",postprocess= True,output_schema=None,logging_schema=None,fn_guard=False,fp_guard=False,voxel_kwargs=None,ep_seed=None,log_oracle=False,
                 explr_bonus = None,
                 collision_penalty = None
                 ):
        from habitat.config.default import get_config
        from habitat.config import read_write
        from habitat.config.default_structured_configs import (
            TopDownMapMeasurementConfig,
            FogOfWarConfig,
        )
        import utils.measures #register custom measures
        from habitat import make_dataset
        import utils.ovon.ovon_dataset
        import utils.ovon.ovon_nav
        # try:
        #     import ovon.dataset.ovon_dataset #register ovon dataset
        #     import ovon.measurements.nav # register ovon measurements
        # except ImportError:

        #     print("OVON not installed, skipping OVON dataset/measurements registration")
        #     pass
        """
        assigned_episode_labels: List of strings ['scene_id_episode_id', ...] specific to this worker.
        enable_caching: If True, stores observations for video generation.
        """
        if workspace is not None:
            # nuclear option to ensure proper habitat loading
            import os
            os.chdir(workspace)
        self.explr_bonus = explr_bonus
        self.collision_penalty = collision_penalty

        self.postprocess = postprocess
        self.log_oracle = log_oracle
        self.enable_caching = enable_caching
        if enable_caching:
            self.steps = defaultdict(list)

        self.logging_schema = logging_schema
        self.output_schema = output_schema

        self.logging_actor = None

        self.fn_guard = fn_guard
        self.fp_guard = fp_guard

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
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5, fov=79),
                )
            )

        # Create Dataset
        self.full_dataset = make_dataset(
            self.config_env.habitat.dataset.type, config=self.config_env.habitat.dataset
        )
        self.ep_seed = ep_seed
      

        if assigned_episode_labels is not None:
            self.assign_shard(assigned_episode_labels)
        else:
            print("skipping sim initialization since no shards provided, please call assign_shard")
            self.episode_counter = 0
            self.shard_length = 0

    def _resolve_task_specific_logic(self):
        """
        rigorously detects the TaskType by inspecting the observation space 
        and configuring the worker's goal abstraction layer.
        """
        # 1. Safe Observation Space Access
        if hasattr(self.env, "observation_space"):
            obs_space = self.env.observation_space
        elif hasattr(self.env, "habitat_env"):
            obs_space = self.env.habitat_env.observation_space
        else:
            print("[Worker] Warning: No observation space found. Defaulting to GENERIC.")
            self.task_type = TaskType.GENERIC
            self._get_instr_or_goal = lambda obs: "N/A"
            return

        self.sensors = obs_space.spaces.keys()
        if hasattr(self.env, "habitat_env"):
            self.measures = self.env.habitat_env.task.measurements.measures
        else:
            self.measures = self.env.unwrapped.habitat_env.task.measurements.measures
        # 2. Strict Detection Logic
        if "objectgoal" in self.sensors:
            self.task_type = TaskType.OBJECTNAV
            self._goal_key = "objectgoal"
            self._setup_objectnav_mapping()

        elif "instruction" in self.sensors or "rxr_instruction" in self.sensors:
            self.task_type = TaskType.VLN
            self._goal_key = "instruction" if "instruction" in self.sensors else "rxr_instruction"
            # VLN goals are just the instruction text (or tokens)
            self._get_instr_or_goal = self._format_vln_goal

        elif "imagegoal" in self.sensors or "instance_imagegoal" in self.sensors:
            self.task_type = TaskType.IMAGENAV
            self._goal_key = "imagegoal"
            self._get_instr_or_goal = lambda obs: "[Pixel Goal]"

        else:
            self.task_type = TaskType.GENERIC
            self._goal_key = None
            self._get_instr_or_goal = lambda obs: "PointGoal" if "pointgoal_with_gps_compass" in self.sensors else "N/A"

        print(f"[Worker] Task Resolved: {self.task_type.name} (Goal Key: {self._goal_key})")

    # --- Type-Specific Helpers (Cleaner than messy Lambdas) ---
    def _setup_objectnav_mapping(self):
        """Pre-computes the ID->Name mapping for ObjectNav to avoid lag."""
        try:
            dataset = self.env.habitat_env._dataset
            if hasattr(dataset, "category_to_task_category_id"):
                mapping = dataset.category_to_task_category_id
                # Invert: {'chair': 0} -> {0: 'chair'}
                self._id_to_name = {v: k for k, v in mapping.items()}
                
                self._get_instr_or_goal = lambda obs: self._id_to_name.get(
                    obs['objectgoal'].item() if hasattr(obs['objectgoal'], 'item') else obs['objectgoal'][0],
                    f"Obj {obs['objectgoal']}"
                )
            else:
                raise("failed to map object id to goal name")
        except (AttributeError, KeyError):
            self._get_instr_or_goal = lambda obs: f"ObjID: {obs['objectgoal']}"

    def _format_vln_goal(self, obs):
        """Safe formatter for R2R/RxR instructions."""
        val = obs[self._goal_key]
        # If it's a string, return it. If it's token IDs, return placeholder.
        if isinstance(val, str):
            return val
        elif isinstance(val,dict):
            return val['text']
        else:
            raise TypeError(f"Instruction must be str, got {type(val)}")
    def total_episodes(self):
        return self.full_dataset.num_episodes
    def assign_shard(self,assigned_episode_labels = None):
        from habitat.core.dataset import EpisodeIterator
        import utils.measures
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
            self._resolve_task_specific_logic() # dependent on the env, so use here
        self.nav_oracle = ObjectNavOracle(
            self.env.habitat_env,
            success_distance=self.config_env.habitat.task.measurements.success.success_distance, 
        )
        # self.env = make_gym_from_config(self.config_env,dataset)
        if self.ep_seed is not None:
            seed = self.ep_seed
        else:
            np.random.seed(os.getpid())
            seed = np.random.randint(0,100000)
            print(f"using seed: {seed}")
        # Setup Iterator 
        self.env.habitat_env.episode_iterator = EpisodeIterator(
            dataset.episodes,
            cycle=True,
            shuffle=True,
            group_by_scene=False,
            seed=seed,
            max_scene_repeat_episodes=4
        )
        if self.output_schema is None:
            self.output_schema = self._generate_auto_schema(True)
        if self.logging_schema is None:
            self.logging_schema = self._generate_auto_schema(True)
        self.summary_schema=self._generate_auto_schema()['info']

        print(f"Actor assigned with shard of {len(dataset.episodes)} episodes.")


    def step(self, action:int,supplementary_logs={}):
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

        extras = {'+fp_stop':-99999*int(not self.fp_guard),'+fn_stop':-99999*int(not self.fn_guard)} # massive neg for not enabled, 0 for enabled but not triggerd.
        # oracle stop guards useful for reducin eval noise. #TODO: use habitat config instead of magic number
        try:
            last_distance = self.last_step['info']['distance_to_goal']
            success_distance = self.config_env.habitat.task.measurements.success.success_distance
            if action==0 and last_distance > success_distance:
                if self.fp_guard:
                    action = np.random.choice([1,2,3]) #chose random non stop action
                extras['+fp_stop'] = 1 #record the false positive incident
            if last_distance<success_distance and action!=0:
                if self.fn_guard: 
                    action = 0
                extras['+fn_stop'] = 1
        except:
            print("warning! cannot calculate oracle guard!")
        import time
        # t0 = time.time()
        obs, reward, done, info = self.env.step(action)      
        # print(f"stepping env took {time.time()-t0}")   
        # t0 = time.time()

        step_dict = {
        "obs":obs,
        "reward": reward,
        "done": done,
        "info": info
        }

        if self.postprocess:
            step_dict = self._postprocess_step(step_dict)
            extras['+stuck'] = np.linalg.norm(np.array(self.last_step['info']['+pos_rots'])-np.array(step_dict['info']['+pos_rots']))<1e-5 # record collisions
            if extras['+stuck'] and self.collision_penalty is not None:
                print("applying collision penalty")
                step_dict['reward']-=self.collision_penalty #0.05
            curr_explored_area = step_dict['info']['top_down_map']['fog_of_war_mask'].sum()
            last_explored_area = self.last_step['info']['top_down_map']['fog_of_war_mask'].sum()
            step_dict['+exploration_delta']=curr_explored_area-last_explored_area
            print("applying exploration bonus")
            if step_dict['+exploration_delta']>0 and self.explr_bonus is not None:
                step_dict['reward']+=self.explr_bonus
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
            
            self._cache_step(self._apply_schema(step_dict | extras,self.logging_schema) | supplementary_logs )
            self.steps['action'].append(action)
        self.last_step = step_dict
        # print(f"postprocessing took {time.time()-t0}")
        return self._apply_schema(step_dict | extras,self.output_schema)

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
        if self.postprocess:
            step_dict = self._postprocess_step(step_dict)
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

    def is_exhausted(self):
        return self.episode_counter >= self.shard_length

    def _postprocess_step(self,step_dict):
        obs = step_dict['obs']
        info = step_dict['info']
        obs['+instr_or_goal'] = self._get_instr_or_goal(obs)
        obs['+curr_step'] = len(self.steps['action'])
        if self.env.habitat_env.sim:
            state = self.env.habitat_env.sim.get_agent_state()
            pos = state.position  # Vector3
            rot = state.rotation  # Quaternion

            info['+pos_rots'] = [
                float(pos[0]), float(pos[1]), float(pos[2]),      # x, y, z
                float(rot.x), float(rot.y), float(rot.z), # qx, qy, qz
                float(rot.w)                                 # w
            ]
            if self.voxel_kwargs is not None:
                from utils.bev_utils import get_patch_coords
                H,W = step_dict['obs']['depth'].shape[:2] 
                obs['+patch_coords'] = get_patch_coords(np.array([info['+pos_rots']]),step_dict['obs']['depth'].reshape((1,H,W)),**self.voxel_kwargs)[0] # 1 by H by W by 3 patch coords
        info['+scene_id']=get_scene_id(self.env.current_episode().scene_id)
        info['+episode_id']=self.env.current_episode().episode_id
        info['+episode_label']=f"{info['+scene_id']}_{info['+episode_id']}"
        curr_idx = self.episode_counter-1
        info['+epoch']= curr_idx // self.shard_length
        info['+step_in_epoch']=curr_idx % self.shard_length
        
        try:
            if self.log_oracle:
                oracle_action = self.nav_oracle.get_best_action()
            else:
                oracle_action = 0
        except:
            oracle_action = -1 # failed to get oracle action
        distance = step_dict['info']['distance_to_goal']
        success_distance = self.config_env.habitat.task.measurements.success.success_distance
        if oracle_action == 0:
            if distance is None or distance > success_distance:
                oracle_action = -1 # navmesh failure cause oracle false stop
        info['+oracle_action'] = oracle_action

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
        # return apply_schema_permissive(data,schema)
        return apply_schema(data,schema)
    
    def _generate_auto_schema(self, include_arrays=False):
        """
        Introspects the environment to build a schema automatically.
        
        Policy:
        - Obs: Drop all tensors > 1D (RGB, Depth, Maps) unless include_arrays=True.
                Keep all scalars/vectors (GPS, Compass, Instructions).
        - Info: Keep all registered metrics (SPL, Success) EXCEPT 'top_down_map'.
        """
        # 1. Introspect Observations
        # Handle the Gym wrapper layers to get the real space
        if hasattr(self.env, "observation_space"):
            obs_space = self.env.observation_space
        elif hasattr(self.env, "habitat_env"):
            obs_space = self.env.habitat_env.observation_space
            
        obs_schema = {}
        for sensor_name, space in obs_space.spaces.items():
            # Heuristic: If it has 2+ dims (H, W, C), it's likely an image -> Drop it.
            # If it is 1D (D,) or Scalar (), it's likely metadata -> Keep it.
            try:
                is_heavy_tensor = len(space.shape) >= 2
                
                if is_heavy_tensor and not include_arrays:
                    obs_schema[sensor_name] = False
                else:
                    obs_schema[sensor_name] = True
            except:
                obs_schema[sensor_name] = True
                # print(f"failed with {sensor_name}")
        # 2. Introspect Metrics (Info)
        info_schema = {}
        # We need access to the underlying Habitat Task to see registered measures
        # Try different access patterns depending on if we are wrapped
        try:
            if hasattr(self.env, "habitat_env"):
                measures = self.env.habitat_env.task.measurements.measures
            else:
                measures = self.env.unwrapped.habitat_env.task.measurements.measures
            
            for measure_name in measures.keys():
                # Special Case: TopDownMap is a heavy dictionary containing a large array.
                # Default to False to save bandwidth.
                if measure_name == "top_down_map" and not include_arrays:
                    info_schema[measure_name] = False 
                else:
                    info_schema[measure_name] = True
        except AttributeError:
            # Fallback if we can't reach the task (unlikely in standard Habitat)
            print("[AutoSchema] Warning: Could not detect metrics. Defaulting to empty info.")

        # 3. Construct Final Schema
        return {
            "obs": {**obs_schema,"instr_or_goal": True},
            "info": info_schema,
            # Always keep these primitives
            "action": True,
            "reward": True,
            "done": True,
            "timestamp": True,
            "fp_stop": True,
            "fn_stop": True,
            "stuck": True
        }
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
        auto_flush = True,
        **kwargs
    ):
        from pathlib import Path
        self.log_dir = Path(logging_output_dir).resolve()

        # Use the new default_logging_schema provided in your prompt if none is passed
            
        super().__init__(*args, **kwargs)
        self.auto_flush = auto_flush
        self.logger_actor = logger_actor
        os.makedirs(self.log_dir, exist_ok=True)

    def _flush_logs_to_disk(self,clear_steps = True):
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
        ep_label = first_info.get('episode_label', f"ep_unknown_{int(time.time())}")
        ep_label = f"{ep_label}.{os.getpid()}@{time.time()}"
        # Create dedicated folder: /logs/scene_X/episode_label/
        # This keeps artifacts (video, thumb, json) grouped together
        save_dir = os.path.join(self.log_dir, scene_id, str(ep_label))
        os.makedirs(save_dir, exist_ok=True)

        # 2. Calculate Metrics & Split Data
        # Returns scalars (episode_logs) and raw lists (sequence_logs)
        episode_logs, sequence_logs = habitat_logging_helper(self.steps,self.summary_schema)
    
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
        seq_path = os.path.join(save_dir, "sequence.json")
        ep_path = os.path.join(save_dir,"summary.json")

        episode_logs["vid/episode_video"] = vid_path  # Path for WandB Video
        episode_logs["raw/trace"] =  seq_path          # Path for WandB Artifact/File
        episode_logs["img/thumbnail"] = thumb_path    # Path for WandB Image
       
        episode_logs |= {
             "worker_pid": os.getpid(),
            "node": os.uname().nodename,
            "timestamp": time.time()
        }
        # 4. Save Raw Sequence Data (The "Raw Data" Fix)
        # Dump trajectory/actions to JSON so we don't lose granular history
        # Convert numpy types to native python for JSON serialization
        def default_serializer(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)

        with open(seq_path, 'w') as f:
            json.dump(sequence_logs, f, default=default_serializer,indent=4)

        with open(os.path.join(self.log_dir,f"results_{os.getpid()}"),'a') as f:
            f.write(json.dumps(episode_logs,default=default_serializer)+"\n") #save the result

        with open(ep_path, 'w') as f:
            json.dump(episode_logs, f, default=default_serializer,indent=4) # append to cumulative logs
            # f.write(json.dumps(episode_logs,default=default_serializer)+"\n") #save the result

        # 5. Construct Global Payload (Paths + Scalars)
        # Merge the scalar metrics with the file paths
        payload = {
            **episode_logs,           
        }
        if clear_steps:
            self.steps = defaultdict(list)  # Clear video cache for new episode

        # 6. Send to Global Logger
        if self.logger_actor:
            import ray
            try:
                ray.get(self.logger_actor.log_row.remote(row=payload),timeout=1.0) 
            except Exception as e:
                print(f"having issue with logging ack! {e}")
                # remove ray.get!
        return ep_path

    def reset(self, episode_id=None,output_schema=None,logging_schema=None):
        if len(self.steps['action'])>0 and self.auto_flush:
            self._flush_logs_to_disk()
        return super().reset(episode_id,output_schema,logging_schema)

    def assign_shard(self, assigned_episode_labels=None):
        if len(self.steps['action'])>0 and self.auto_flush:
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

