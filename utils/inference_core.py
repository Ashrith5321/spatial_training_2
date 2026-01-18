import ray
from collections import deque
from typing import List, Dict, Any, Iterator,Tuple
from string import Template
from PIL import Image
from utils.habitat_worker import LoggingHabitatWorker
from utils.vlm_worker import VLMWorker
import numpy as np
from tqdm import tqdm
from utils.prompt_utils import substitute_convo_template, prepare_sequence_inputs


class EpisodeRolloutMixin:
    def run_episode(self,habitat_handle, initial_state_ref):
        import time
        try:
            # 1. Resolve the initial state (Blocking wait for reset to finish)
            # Ray automatically waits for initial_state_ref to be ready before starting this task,
            # but we call ray.get to access the data.
            pos_id_kwargs={
                "mode": "standard" #TODO: make this configurable
            }
            if len(initial_state_ref)==2:
                rgb,state_dict = initial_state_ref
            elif len(initial_state_ref)==3:
                rgb,patch_coords,state_dict = initial_state_ref
                pos_id_kwargs['patch_coords'] = patch_coords
                pos_id_kwargs['mode'] = "bev"
            step_count = 0
            done = False
            
            if self.debug_with_humanaction:
                human_data = ray.get(habitat_handle.get_human_data.remote())
                seq_messages, seq_images = prepare_sequence_inputs(human_data, state_dict['obs'] | self.rollout_config)
                seq_action_probs = self.infer_sequence(seq_messages, seq_images, human_data, temperature = self.rollout_config['temperature'],pos_id_kwargs=pos_id_kwargs)
                self.reset()
            messages = substitute_convo_template(self.rollout_config['convo_start_template'],state_dict['obs'] | self.rollout_config)
            # 2. The Interaction Loop
            vlm_logs={}
            goal_name = state_dict['obs']['goal_name']
            while not done and step_count < self.rollout_config['max_steps']:
                # A. Prepare VLM Input
                # (Assuming your formatting logic is here)
                if isinstance(rgb, np.ndarray):
                    rgb_pil = Image.fromarray(rgb)
                else:
                    rgb_pil = rgb
                # B. Call VLM (Blocking)
                # We must wait for the answer to decide the next step
                # print("inferring VLM with messages:")
                # print(messages)
                t0 = time.time()
                action_probs = self.infer_probs(images=[rgb_pil],messages=messages,temperature = self.rollout_config['temperature'],pos_id_kwargs=pos_id_kwargs)
                vlm_logs |= {'mean/vlm_latency':time.time()-t0,'min/vlm_latency':time.time()-t0,'max/vlm_latency':time.time()-t0}
                # print(f"vlm step{step_count}")
                # print("done")
                #except for the first turn, all messages follow the exact same template.
                action_id = np.random.choice(len(action_probs),p=action_probs) # sampling
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
                vlm_logs |= {'mean/entropy':entropy,'mean/action_prob':action_probs[action_id]} 
                del rgb,state_dict

                # D. Step Simulator (Blocking) ---------------------------RAY----------------------------- 
                t0 = time.time()
                state_ref = ray.get(habitat_handle.step.remote(action_id,step_count,supplementary_logs=vlm_logs))
                if self.debug_with_humanaction:
                    action_id = ray.get(habitat_handle.get_step_data.remote())['action'][-1]
                if len(state_ref)==2:
                    rgb,state_dict = state_ref
                elif len(state_ref)==3:
                    rgb,patch_coords,state_dict = state_ref
                    pos_id_kwargs['patch_coords'] = patch_coords
                    pos_id_kwargs['mode'] = "bev"
                vlm_logs = {'mean/sim_latency':time.time()-t0,'min/sim_latency':time.time()-t0,'max/sim_latency':time.time()-t0}
                messages = substitute_convo_template(self.rollout_config['convo_turn_template'],state_dict['obs'] | {"action":self.rollout_config['action_space'][action_id]})
                # print(f"sim step{step_count}")
                done = state_dict['done']
                step_count += 1
            if self.debug_with_humanaction:
                self.compare_with_seq()
            return ray.get_runtime_context().current_actor,habitat_handle,state_dict['is_exhausted'], state_dict['info'] | {"steps":step_count, "goal_name":goal_name}
        except Exception as e:
            print(f"Episode failed: {e}")
            import traceback
            traceback.print_exc()
            # Return handles anyway so we don't leak resources (or handle crash logic)
            return ray.get_runtime_context().current_actor,habitat_handle,False, None
        finally:
            self.reset()

# from utils.logging_worker import LoggingHabitatWorker

@ray.remote
class HabitatRayWorker(LoggingHabitatWorker):
    """
    Ray Actor wrapper for the LoggingHabitatWorker.
    
    Optimizations:
    1. Interface Shaping: Separates heavy RGB arrays from lightweight scalar state.
    2. RPC Reduction: Injects 'is_exhausted' into the state dictionary.
    """

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns:
            rgb: The heavy image array.
            state_dict: {'obs': ..., 'is_exhausted': ...}
        """
        # Base worker returns a single dict (typically just the observations for reset)
        obs = super().reset()['obs']
        
        # 1. Extract the heavy asset (modifies dict in place)
        rgb = obs.pop("rgb")
        
        # 2. Package the lightweight state
        # We wrap it in an 'obs' key to match the structure expected by the Mixin
        state_dict = {
            "obs": obs,
            "is_exhausted": self.is_exhausted(),
            # Optional: Add placeholders if your Mixin checks these on reset
            "done": False, 
            "info": {}
        }
        
        if self.voxel_kwargs is None:
            return rgb, state_dict
        else:
            patch_coords = state_dict['obs'].pop('patch_coords')
            return rgb,patch_coords,state_dict


    def step(self, action: int, step_count: int, supplementary_logs: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns:
            rgb: The heavy image array.
            state_dict: {'obs': ..., 'reward': ..., 'done': ..., 'info': ..., 'is_exhausted': ...}
        """
        # Base worker returns a single dict containing keys: 'obs', 'reward', 'done', 'info'
        result = super().step(action, step_count, supplementary_logs=supplementary_logs)
        # 1. Extract the heavy asset from the nested observation dict
        # We modify the dictionary in-place to avoid copying data
        rgb = result['obs'].pop("rgb")
        # 2. Inject the exhaustion flag
        result['is_exhausted'] = self.is_exhausted()
        # 'result' now acts as our 'state_dict' (sans the heavy RGB)
        if self.voxel_kwargs is None:
            return rgb, result
        else:
            patch_coords = result['obs'].pop('patch_coords')
            return rgb,patch_coords,result

@ray.remote
class VLMRayWorker(VLMWorker, EpisodeRolloutMixin):
    def __init__(self, rollout_config: Dict[str, Any], **vlm_kwargs):
        """
        Explicitly handles argument separation to avoid MRO issues.
        
        Args:
            rollout_config: Arguments intended for the EpisodeRolloutMixin.
            **vlm_kwargs: All other arguments (model_id, dtype, etc.) passed to VLMWorker.
        """
        # 1. Initialize the VLM Worker (The Heavy Lifter)
        # We pass only the relevant VLM args to avoid 'unexpected keyword argument' errors.
        VLMWorker.__init__(self, **vlm_kwargs)
        
        # 2. Initialize the Mixin State
        # Since the Mixin's __init__ was just setting this variable, we can do it here directly
        # effectively bypassing the need for cooperative inheritance in the parents.
        self.rollout_config = rollout_config

def run_inference_driver(
    sim_handles: List[Any],
    vlm_handles: List[Any],
    shard_iterator: Iterator[List[str]]
) -> List[Dict]:
    """
    Orchestrates the evaluation pipeline.

    Args:
        sim_handles: List of Ray actor handles for Habitat workers.
        vlm_handles: List of Ray actor handles for VLM workers.
        config: Configuration dict passed to the supervisor.
        shard_iterator: An iterator yielding lists of episode IDs (shards).
    """

    # --- 1. Initialize Pools ---
    # Idle VLMs: Ready to be assigned immediately
    idle_vlms = deque(vlm_handles)
    
    # Ready Habitats: Tuple of (actor_handle, initial_state_ref)
    # These are workers that have finished resetting and are waiting for a VLM.
    ready_habitats = deque()
    # --- 2. Tracking Futures (The State Machine) ---
    # Map: reset_future -> habitat_handle
    # Tracks workers currently resetting (loading scene or moving to next episode).
    pending_resets = {} 
    # Map: supervisor_future -> "metadata"
    # Tracks active episodes running in the background.
    active_episodes = {}
    # Collection of all results
    results = []
    # --- 3. Bootstrap: Initial Sharding & Resets ---
    print(f"Bootstrapping: Initializing {len(sim_handles)} environments...")
    
    # We must assign an initial shard to every habitat worker before they can reset.
    for sim_handle in sim_handles:
        try:
            # Assign first shard
            initial_shard = next(shard_iterator)
            sim_handle.assign_shard.remote(initial_shard)
            
            # Trigger first reset
            # The worker will load the first episode in the shard.
            reset_ref = sim_handle.reset.remote()
            pending_resets[reset_ref] = sim_handle   
        except StopIteration:
            print("Warning: Not enough shards for all workers during bootstrap.")
            # Worker is retired immediately if no work exists
            pass
    # --- 4. The Event Loop ---
    # We run as long as there is active work (resets or episodes) or potential work.
    
    while active_episodes or pending_resets or (ready_habitats and idle_vlms):
        
        # A. Check for "Ready to Pair" Condition
        # If we have an idle VLM and a ready Habitat, launch a task immediately.
        while idle_vlms and ready_habitats:
            vlm = idle_vlms.popleft()
            hab, init_state_ref = ready_habitats.popleft()
            
            # LAUNCH SUPERVISOR
            # The supervisor coordinates the interaction between VLM and Habitat for ONE episode.
            print("dispatching new episode!")
            sup_ref = vlm.run_episode.remote(
                hab, init_state_ref
            )
            
            active_episodes[sup_ref] = "running"

        # B. Wait for SOMETHING to happen
        # We listen to both pool (resets) and active tasks (episodes).
        all_watch_refs = list(pending_resets.keys()) + list(active_episodes.keys())
        
        if not all_watch_refs:
            # Only happens if iterator exhausted and all workers are idle (shutdown)
            break

        # Blocking wait for the FIRST completed future to maximize responsiveness
        ready_refs, _ = ray.wait(all_watch_refs, num_returns=1)
        
        for ref in ready_refs:
            
            # --- CASE 1: A Habitat Finished Resetting ---
            if ref in pending_resets:
                sim_handle = pending_resets.pop(ref)
                print("new habitat worker ready!",end=" ")
                # The worker is now ready for a VLM.
                # We store the ref (initial observation) to pass to the supervisor.
                ready_habitats.append((sim_handle, ref))
            
            # --- CASE 2: An Episode Finished ---
            elif ref in active_episodes:
                del active_episodes[ref]
                
                # Retrieve the recycled actors and signals
                # needs_reshard: bool indicating if the worker finished its assigned shard
                vlm, hab, needs_reshard, stats = ray.get(ref)
                print(f"[new results!] :{stats}")
                results.append(stats)
                # 1. Recycle VLM (It becomes idle immediately)
                idle_vlms.append(vlm)
                
                # 2. Recycle Habitat
                try:
                    if needs_reshard:
                        print("worker depleted! trying to assigning new shard")
                        # Pull new work from the iterator
                        new_shard = next(shard_iterator)
                        hab.assign_shard.remote(new_shard)
                    # Whether we got a new shard or are continuing the old one,
                    # we must reset to prepare the next episode.
                    new_reset_ref = hab.reset.remote()
                    pending_resets[new_reset_ref] = hab
                except StopIteration:
                    ray.get(hab._flush_logs_to_disk.remote())
                    # No more work available. Retire the Habitat worker.
                    print(f"Worker finished and no shards remain. Retiring.")
                    pass
    print(f"Inference complete. Processed {len(results)} episodes.")
    print(f"Cleaning up by forcing log flush...")
    for sim_handle in tqdm(sim_handles):
        ray.get(sim_handle._flush_logs_to_disk.remote())
    print("done flushing!")
    return results