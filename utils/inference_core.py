import ray
from collections import deque
from typing import List, Dict, Any, Iterator,Tuple
from string import Template
from PIL import Image
from utils.habitat_worker import LoggingHabitatWorker
from utils.vlm_worker import VLMWorker,VLMTrainingMixin
import numpy as np
from tqdm import tqdm
from utils.tensor_utils import TensorPacker

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
    
def substitute_convo_template(conversation_template: List[Dict], substitutions: Dict[str, Any]) -> List[Dict]:
    """
    Traverses the conversation template and substitutes any string.Template 
    objects found in 'text' fields using values from the 'obs' dictionary.
    
    Args:
        conversation_template: List of message dicts (role, content).
        substitutions: Dictionary containing substitution keys (e.g., 'instr_or_goal').
        
    Returns:
        A new conversation list with strings substituted.
    """
    new_conversation = []
    
    for message in conversation_template:
        # Shallow copy the message container
        new_message = message.copy()
        new_content = []
        
        # Iterate over the content list (e.g., [{"type": "image"}, {"type": "text", ...}])
        for item in message.get("content", []):
            new_item = item.copy()
            
            # Check if this item is a text component
            if "text" in new_item:
                text_obj = new_item["text"]
                
                # CASE A: It's a Template object (from the config)
                if "$" in text_obj:
                    try:
                        text_template = Template(text_obj)
                        # Perform the substitution
                        new_item["text"] = text_template.substitute(substitutions)
                    except KeyError as e:
                        raise
                        # Fallback to safe_substitute to prevent crashing on missing keys,
                        # but log it so we know something is wrong.
                        # print(f"Warning: Missing substitution key {e} in template.")
                        # new_item["text"] = text_template.safe_substitute(substitutions)
                        
                # CASE B: It's already a str (static text)
                elif isinstance(text_obj, str):
                    pass # Keep as is
                    
            new_content.append(new_item)
        new_message["content"] = new_content
        new_conversation.append(new_message)
        
    return new_conversation

class EpisodeRolloutMixin:
    def _pack_trajectory(self, buffer: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Converts list of dicts to a dict of numpy arrays (Columnar format).
        This format allows Ray to zero-copy transfer individual columns.
        """
        if not buffer:
            return {}
        
        # Fast dictionary of lists to list of dictionaries inversion
        keys = buffer[0].keys()
        stacked = {k: np.array([d[k] for d in buffer]) for k in keys}
        
        # Optimization: Cast probabilities to float32 to save 50% bandwidth
        if "probs" in stacked:
            stacked["action_probs"] = stacked["action_probs"].astype(np.float32)
        if "rewards" in stacked:
             stacked["rewards"] = stacked["rewards"].astype(np.float32)    
        return stacked
    
    def run_episode(self,habitat_handle, initial_state_ref,collect_trajectory=False,compute_value=False):
        import time
        ep_t0 = time.time()
        self.reset() #we reset at the start to ensure clean state. not resetting at the end preserves state for downstream.
        try:
            # 1. Resolve the initial state (Blocking wait for reset to finish)
            # Ray automatically waits for initial_state_ref to be ready before starting this task,
            # but we call ray.get to access the data.
            pos_id_kwargs={
                "mode": "standard"
            }
            if len(initial_state_ref)==2:
                rgb,state_dict = initial_state_ref
                print("using normal mode")
            elif len(initial_state_ref)==3:
                print("using bev mode")
                rgb,patch_coords,state_dict = initial_state_ref
                pos_id_kwargs['patch_coords'] = patch_coords
                pos_id_kwargs['mode'] = "bev"
            step_count = 0
            done = False
            messages = substitute_convo_template(self.rollout_config['convo_start_template'],state_dict['obs'] | self.rollout_config)
            # 2. The Interaction Loop
            vlm_logs={}
            # Trajectory Buffer (List is fine here!)
            trajectory_buffer = []
            instr_or_goal = state_dict['obs']['instr_or_goal']
            episode_label = state_dict['info']['episode_label']
            while not done and step_count < self.rollout_config['max_steps']:
                # A. Prepare VLM Input
                rgb_numpy = rgb
                rgb_pil = Image.fromarray(rgb_numpy)
                # B. Call VLM (Blocking)
                # We must wait for the answer to decide the next step
                # print("inferring VLM with messages:")
                # print(messages)
                t0 = time.time()
                action_probs,action_logprobs,outputs = self.infer_probs(images=[rgb_pil],messages=messages,temperature = self.rollout_config['temperature'],pos_id_kwargs=pos_id_kwargs)
                
                vlm_logs |= {'mean/vlm_latency':time.time()-t0,'min/vlm_latency':time.time()-t0,'max/vlm_latency':time.time()-t0,'sum/spguard_trigger_count':0}
                try:
                    import torch
                    vlm_logs |= {"vlm_mem_GB":torch.cuda.memory_allocated()/(1024**3)}
                except:
                    print("warning: could not get vlm mem")
                # print(f"vlm step{step_count}")
                # print("done")
                #except for the first turn, all messages follow the exact same template.
                action_id = np.random.choice(len(action_probs),p=action_probs) # sampling
                if action_id ==0 and self.rollout_config['stop_prob_threshold'] is not None:
                    if action_probs[0] >= self.rollout_config['stop_prob_threshold']:
                        action_id = 0
                    else:
                        vlm_logs['sum/spguard_trigger_count']=1
                        action_id = np.random.choice(len(action_probs)-1,p=action_probs[1:]/np.sum(action_probs[1:]))+1
                    
                oracle_action_id = state_dict['info'].get('oracle_action',-1) # only update temporally afterwards

                entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
                vlm_logs |= {'mean/entropy':entropy,'mean/action_prob':float(action_probs[action_id]),"action_probs":action_probs.tolist()} 
                vlm_logs |= {'mean/token_keep_fraction':float(self.seq_keep_mask.float().mean())}
                vlm_logs |= {'mean/num_tokens':int(self.seq_keep_mask.shape[-1])}

                # D. Store Transition
               

                # D. Step Simulator (Blocking) ---------------------------RAY----------------------------- 
                t0 = time.time()
                # del rgb,state_dict
                state_ref = ray.get(habitat_handle.step.remote(action_id,supplementary_logs=vlm_logs))
                if len(state_ref)==2:
                    rgb,state_dict = state_ref
                elif len(state_ref)==3:
                    rgb,patch_coords,state_dict = state_ref
                    pos_id_kwargs['patch_coords'] = patch_coords
                    pos_id_kwargs['mode'] = "bev"
                vlm_logs = {'mean/sim_latency':time.time()-t0,'min/sim_latency':time.time()-t0,'max/sim_latency':time.time()-t0}
                if collect_trajectory:
                    # Append dict to list - fast and simple
                    trajectory_dict = {
                        "actions": action_id,
                        "rollout_logprobs": action_logprobs,
                        "rollout_probs": action_probs,
                        "rewards": state_dict.get("reward", 0.0),
                        "dones": state_dict['done'],
                        "spl": state_dict['info']['spl'],
                        "success": state_dict['info']['success'],
                        "distance_to_goal": state_dict['info']['distance_to_goal'],
                        "oracle_actions": oracle_action_id
                    }
                    if compute_value:
                        import torch
                        # Compute value estimate for the current state
                        with torch.no_grad():
                            trajectory_dict["values"] = self._compute_value(outputs).cpu().numpy()
                    trajectory_buffer.append(trajectory_dict)
                messages = substitute_convo_template(self.rollout_config['convo_turn_template'],{"action":self.rollout_config['action_space'][action_id]})
                # print(f"sim step{step_count}")
                done = state_dict['done']
                step_count += 1
                # Convert list of dicts -> Dict of Numpy Arrays (Zero-Copy Friendly)
            final_trajectory = self._pack_trajectory(trajectory_buffer) if collect_trajectory else None
            final_info = state_dict['info'] | {"steps":step_count, "instr_or_goal":instr_or_goal}
            # Return Clean Tuple (No Actor Handles here)
            final_info['episode_duration'] = time.time()-ep_t0
            return habitat_handle, state_dict['is_exhausted'], final_info, final_trajectory
        
        except Exception as e:
            print(f"Episode failed: {e}")
            import traceback
            traceback.print_exc()
            # Return handles anyway so we don't leak resources (or handle crash logic)
            return habitat_handle,False, None,None

class RolloutWorker(VLMWorker, EpisodeRolloutMixin):
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
        import os
        np.random.seed(os.getpid())
        # 2. Initialize the Mixin State
        # Since the Mixin's __init__ was just setting this variable, we can do it here directly
        # effectively bypassing the need for cooperative inheritance in the parents.
        self.rollout_config = rollout_config

class InferenceRayWorker(RolloutWorker):
    def run_episode(self,habitat_handle, initial_state_ref):
        habitat_handle,is_exhausted,final_info,_ = super().run_episode(habitat_handle, initial_state_ref)
        return ray.get_runtime_context().current_actor,habitat_handle,is_exhausted,final_info

class RLWorker(RolloutWorker,VLMTrainingMixin):
    def __init__(self, rollout_config: Dict[str, Any], **vlm_kwargs):
        """
        Combines VLM inference, RL Data Collection, and Training capabilities.
        """
        # 1. Initialize VLM (Heavy weights)
        VLMWorker.__init__(self, **vlm_kwargs)
        
        # 2. Initialize Rollout Config
        self.rollout_config = rollout_config
        
    def run_episode(self,habitat_handle,initial_state_ref,rtn_inputs = False,rtn_embeds=True):
        '''
        stateful run episode. stores the trajectory and sequence level model inputs internally so we can release the habitat ref, and later calculate the logprobs.
        '''
        self.rl_seq_inputs = None
        self.rl_embeds_inputs = None
        self.rl_trajectory = None
        if rtn_inputs:
            self.save_pixels = True #need pixels to reconstruct sequence inputs.
        else:
            self.save_pixels = False
        habitat_handle,is_exhausted,state_dict,trajectory = super().run_episode(habitat_handle, initial_state_ref,collect_trajectory=True,compute_value=False)
        self.rl_trajectory = trajectory
        inputs,embeds = None,None
        if rtn_embeds:
            embeds = self._pack_embeds()
            self.rl_embeds_inputs = embeds
        if rtn_inputs:
            inputs = self._pack_inputs()
            self.rl_seq_inputs = inputs
        return habitat_handle,is_exhausted,state_dict,trajectory,inputs,embeds

    def postprocess_episode(self,eval=False):
        '''
        clears the internal state and returns the processed trajectory and model inputs.
        - trajectory includes: 
            - rollout logprobs (for rollout correction)
            - old logprobs (calculated from same weights as rollout model but with full forward pass instead of kv cache)
        If eval is True, skips forward passes and just returns raw trajectory
        '''
        model_inputs = None

        if not eval: # skip logprobs calculation during eval for speed.
            values = None
            import torch
            with torch.no_grad():
                if self.rl_embeds_inputs is not None:
                    logits,values = self._forward_embeds(self.rl_embeds_inputs,None,self.rl_algo_config.use_value)
                    model_inputs = self.rl_embeds_inputs
                elif self.rl_seq_inputs is not None:
                    logits,values = self._forward_seq(self.rl_seq_inputs,None,self.rl_algo_config.use_value)
                    model_inputs = self.rl_seq_inputs
                else:
                    raise ValueError("No stored model inputs found for postprocessing.")
                logprobs = self._calculate_action_logprobs(logits).squeeze().float().cpu().numpy()
                self.rl_trajectory['old_logprobs'] = logprobs
                if values is not None:
                    self.rl_trajectory['values'] = values.squeeze().float().cpu().numpy()

            if self.rl_algo_config.use_ref:
                with torch.no_grad():
                    self.unmerge_adapter() 
                    with self.model.disable_adapter():
                        if self.rl_embeds_inputs is not None:
                            logits,values = self._forward_embeds(self.rl_embeds_inputs,None,False)
                        elif self.rl_seq_inputs is not None:
                            logits,values = self._forward_seq(self.rl_seq_inputs,None,False)
                        ref_logprobs = self._calculate_action_logprobs(logits).squeeze().float().cpu().numpy()
                        self.rl_trajectory['ref_logprobs'] = ref_logprobs

        return self.rl_trajectory,model_inputs    
    
class RLRayWorker(RLWorker):
    def run_episode(self,habitat_handle,initial_state_ref):
        habitat_handle,is_exhausted,state_dict,_,_,_ = super().run_episode(habitat_handle, initial_state_ref)
        return ray.get_runtime_context().current_actor,habitat_handle,is_exhausted,state_dict
    
    def postprocess_episode(self,return_inputs = True,eval=False):
        trajectory,model_inputs = super().postprocess_episode(eval=eval)
        if return_inputs:
            inputs_tensors,inputs_metadata = TensorPacker.pack(model_inputs)
            return trajectory,inputs_tensors,inputs_metadata
        else: return trajectory,

    def train_rl_step(self, embeds_inputs_np, embeds_inputs_meta,traj_batch):
        embeds_inputs = TensorPacker.unpack(embeds_inputs_np,embeds_inputs_meta,device=self.accelerator.device)
        actions = traj_batch['actions']
        old_log_prob = traj_batch['old_log_prob']
        advantages = traj_batch['advantages']
        returns = traj_batch['returns']
        old_values = traj_batch.get('values',None)
        rollout_log_probs = traj_batch.get('rollout_logprobs',None)
        ref_logprobs = traj_batch.get('ref_logprobs',None)
    
        return super().train_rl_step(embeds_inputs, actions, old_log_prob, advantages, returns, old_values, rollout_log_probs,ref_logprobs)
    
    def train_dagger_step(self, embeds_inputs_np, embeds_inputs_meta, traj_batch):
        """
        Pure DAgger (Behavior Cloning) Step.
        """
        embeds_inputs = TensorPacker.unpack(embeds_inputs_np, embeds_inputs_meta, device=self.accelerator.device)
        
        # 1. Prepare Data
        # Ensure we have the mask (default to all valid tokens if not provided)
        dagger_mask = traj_batch.get('dagger_mask', traj_batch.get('response_mask', None))
        if dagger_mask is None:
            import torch
             # Fallback: Create a ones mask matching action shape
            dagger_mask = torch.ones_like(traj_batch['actions'], dtype=torch.bool)

        expert_actions = traj_batch['oracle_actions']
        # Robustness: Handle One-Hot vs Indices
        # If input is (B, S, A) probabilities, convert to (B, S) indices
        if expert_actions.dim() > 2:
            expert_actions = expert_actions.argmax(dim=-1)

        # 2. Dispatch
        return super().generic_train_step(
            embeds_inputs=embeds_inputs,
            loss_fn_names=['bc'],
            loss_kwargs_list={
                'bc': {
                    'expert_actions': expert_actions,
                    'dagger_mask': dagger_mask
                }
            }
        )

    def train_dagrl_step(self, embeds_inputs_np, embeds_inputs_meta, traj_batch,rl_weight=1.0,bc_weight=0.1):
        """
        Hybrid Step: PPO + DAgger.
        Assumes the driver has populated 'response_mask' (for PPO) and 
        'dagger_mask' (for BC) to separate the data.
        """
        embeds_inputs = TensorPacker.unpack(embeds_inputs_np, embeds_inputs_meta, device=self.accelerator.device)
        
        # 1. Unpack RL Args
        actions = traj_batch['actions']
        old_log_prob = traj_batch['old_log_prob']
        advantages = traj_batch['advantages']
        returns = traj_batch['returns']
        old_values = traj_batch.get('values', None)
        rollout_log_probs = traj_batch.get('rollout_logprobs', None)
        ref_logprobs = traj_batch.get('ref_logprobs', None)
        
        # The driver MUST provide this to avoid double-counting gradients
        # (e.g. response_mask should be FALSE for DAgger samples)
        rl_mask = traj_batch.get('response_mask') 
        if rl_mask is None:
            import torch
             # Fallback (dangerous for hybrid): Assume all valid tokens are RL
            rl_mask = torch.ones_like(actions, dtype=torch.bool)

        # 2. Unpack DAgger Args
        expert_actions = traj_batch['oracle_actions']
        if expert_actions.dim() > 2:
            expert_actions = expert_actions.argmax(dim=-1)
            
        dagger_mask = traj_batch.get('dagger_mask')
        if dagger_mask is None:
            import torch
            # If missing in hybrid, assume NO DAgger (safety default)
            dagger_mask = torch.zeros_like(actions, dtype=torch.bool)

        # 3. Dispatch
        return super().generic_train_step(
            embeds_inputs=embeds_inputs,
            loss_fn_names=['rl', 'bc'],
            loss_kwargs_list={
                'rl': {
                    'actions': actions,
                    'old_log_prob': old_log_prob,
                    'advantages': advantages,
                    'returns': returns,
                    'old_values': old_values,
                    'rollout_log_probs': rollout_log_probs,
                    'ref_log_probs': ref_logprobs,
                    'response_mask': rl_mask
                },
                'bc': {
                    'expert_actions': expert_actions,
                    'dagger_mask': dagger_mask
                }
            },
            loss_weights=[rl_weight, bc_weight]
        )
        
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
        state_dict = super().reset()
        
        # 1. Extract the heavy asset (modifies dict in place)
        rgb = state_dict['obs'].pop("rgb")

        state_dict['is_exhausted'] = self.is_exhausted()
        
        if self.voxel_kwargs is None:
            return rgb, state_dict
        else:
            patch_coords = state_dict['obs'].pop('patch_coords')
            return rgb,patch_coords,state_dict


    def step(self, action: int, supplementary_logs: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns:
            rgb: The heavy image array.
            state_dict: {'obs': ..., 'reward': ..., 'done': ..., 'info': ..., 'is_exhausted': ...}
        """
        # Base worker returns a single dict containing keys: 'obs', 'reward', 'done', 'info'
        result = super().step(action, supplementary_logs=supplementary_logs)
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