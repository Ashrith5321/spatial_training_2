import ray
import os
from PIL import Image
import numpy as np
import sys
import platform
from utils.habitat_worker import HabitatWorker
from utils.vlm_worker import VLMWorker
import textwrap
import time
from misc.constants import episode_labels_table
import json
# Get the IP from environment or hardcode it
head_ip = os.environ.get("HEAD_IP")

if not head_ip:
    print("Error: HEAD_IP environment variable not set.")
    sys.exit(1)

print(f"Connecting to Ray Cluster at {head_ip}:6379...")


# Connect to the existing cluster
ray.init(address=f"{head_ip}:6379",log_to_driver=False)
remoteHabitat =ray.remote(
    num_gpus=0.2,                   # Allocate 1/4th of a GPU
    resources={"env_b": 0.1},)(HabitatWorker)
remoteVLM = ray.remote(
    num_gpus=0.75,                   # Allocate 1/4th of a GPU
    resources={"env_a": 0.1},)(VLMWorker)
output_schema = {
    "obs":{
        "rgb":True,
        "goal_name":True
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
ATTN_IMPL = "flash_attention_2"
DTYPE = "float16"
save_dir = f'/home/huyu/scratch/qixin/dump/{DTYPE}_{ATTN_IMPL}_inference/'
os.makedirs(save_dir,exist_ok=True)

#------ stop oracles ------
guard_stop = True #prevents false positive stops with oracle
auto_stop = False #prevents false negative stops with oracle
#------ action space config
VOCAB = ["stop","forward","left","right"]#,"up","down"] #

habitat_worker = remoteHabitat.remote(workspace='/home/huyu/scratch/qixin/',assigned_episode_labels=episode_labels_table['sample400_a'],output_schema=output_schema)
vlm_worker = remoteVLM.remote(prefix = '<|im_start|>assistant\n**',postfix = '**<|im_end|>',model_id='Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800',attn_implementation=ATTN_IMPL,dtype=DTYPE,vocab = VOCAB)

step_message_template = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, # Placeholder for the pixel data
        ],
    },
    {
        "role": "assistant",
        "content":[
            {"type":"text","text": "**forward**"}
        ]
    }
]

def sample_action(logprobs):
    """
    Samples an action index from a 1D numpy array of log-probabilities.
    
    Args:
        logprobs (np.array): 1D array of shape (num_actions,).
    """

    # Convert to probabilities
    probs = np.exp(logprobs)
    
    # Normalize (ensure sum is exactly 1.0 to satisfy np.random.choice)
    probs /= np.sum(probs)
    
    # Sample
    action_id = np.random.choice(len(probs), p=probs)
    return action_id
try:
    results_f = open(save_dir+'results.txt',mode='a')
    success = 0
    spl = 0
    for j in range(1,201):
            # Do work
        # 2. Get Observation from Habitat (Env B)
        # We use ray.get() here to bring the data to the driver
        # Recall: We sanitized this in the previous step, so it's a pure Python dict/numpy
        curr_state = ray.get(habitat_worker.reset.remote())
        ray.get(vlm_worker.reset.remote())

        rgb_numpy = curr_state['obs']['rgb'] 
        image_obj = Image.fromarray(rgb_numpy)

        system_prompt = textwrap.dedent(f"""\
        You are a visual navigation agent tasked with finding "$goal_text" in an unknown environment.
        You will receive a sequence of observations showing your movement history up to the current moment.

        **Action Space:**
        [stop, forward, left, right, up, down]

        **Your Mission:**
        1. Analyze the observation history to understand your current location and orientation.
        2. Select the next discrete action to navigate efficiently towards the goal.

        **Critical Constraints:**
        * **Collision Detection:** If your previous action was **forward** but the visual observation did not change significantly, you have collided. You MUST turn or move away immediately. Do not keep pushing forward.
        * **Success Condition:** Output **stop** ONLY when the target is plainly in view, centered, and within 1 meter (close enough to touch).

        **Output Format:**
        Respond with the selected action inside double asterisks.
        """)
        # 3. Extract and Process Image
        # Habitat 'rgb' is typically a (H, W, 3) numpy array

        # 4. Construct Message for VLM
        # The instruction asks the VLM to pick the next move

        messages = [
            {
                "role": "user",
                "content": [ # Placeholder for the pixel data
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [ # Placeholder for the pixel data
                    {"type": "image"}
                ],
            },
            {
                "role": "assistant",
                "content":[
                    {"type":"text","text": "**forward**"}
                ]
            }
        ]
        i = 0
        t0 = time.time()
        print("STARTING EPISODE")
        print(f"goal: {curr_state['obs']['goal_name']} episode_label: {curr_state['info']['episode_label']}")
        while not curr_state['done'] and i<300:
                # 5. Invoke VLM (Env A)
            # We pass the PIL image list. Ray serializes it and sends it to Env A.
            logprobs_ref = vlm_worker.infer_step.remote(
                messages=messages, 
                images=[image_obj],
                full_logprobs=False
            )

            # 6. Get Result
            logprobs = ray.get(logprobs_ref)
            # print(f"VLM Output Shape: {logprobs}")

            assert(len(logprobs)==1)
            logprobs = logprobs[0]

            action_id = sample_action(logprobs)
            if action_id == 0 and guard_stop and curr_state['info']['distance_to_goal']>0.1:
                action_id = np.random.choice([1,2,3])
            if auto_stop and curr_state['info']['distance_to_goal']<0.1:
                action_id = 0
            # print(f"stepping simulator with action {action_id}")
            curr_state = ray.get(habitat_worker.step.remote(action_id))

            rgb_numpy = curr_state['obs']['rgb'] 
            image_obj = Image.fromarray(rgb_numpy)
            messages = step_message_template
            i+=1
        success += curr_state['info']['success']
        spl += curr_state['info']['spl']
        elapsed = time.time()-t0
        print(f"done inferring {i} steps in {elapsed} seconds")
        print(f"average throughput: {i/elapsed} steps/s")
        print(f"cumulative SR: {success/j*100}% SPL: {spl/j*100}%")
        result_dict = curr_state['info']
        result_dict['step'] = i
        result_dict['cum_success'] = success/j
        result_dict['cum_spl'] = spl/j
        # print(json.dumps(result_dict)+"\n")
        results_f.write(json.dumps(result_dict)+"\n")
        results_f.flush()
        os.fsync(results_f.fileno())
        ray.get(habitat_worker.save_video.remote(output_dir = save_dir+'videos'))

except Exception as e:
    print(f"Script crashed: {e}")

finally:
    print("Cleaning up actor...")
    results_f.close()
    # Force kill the actor immediately
    ray.kill(habitat_worker)
    ray.kill(vlm_worker)
    # Optional: If you want to be polite and let it close files/logs first:
    # habitat_actor.close.remote() # Assuming you defined a close method
    # ray.kill(habitat_actor)