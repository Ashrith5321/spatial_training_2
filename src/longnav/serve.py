from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
from PIL import Image
import io
import uvicorn
from typing import Optional
from dataclasses import asdict
from longnav.utils.rollout_core import RLWorker
from longnav.config_schema import RolloutConfig,VLMConfig,VLMTrainingConfig
from longnav.utils.factories import resolve_checkpoint_path,get_base_model
from longnav.utils.rollout_core import substitute_convo_template
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
import numpy as np
rollout_cfg = RolloutConfig()
vlm_cfg = VLMConfig()
vlm_cfg.attn_impl = "sdpa"
training_cfg = VLMTrainingConfig()
checkpoint_path = "Aasdfip/hm3d_rpp_ke_standard-checkpoint_231"

checkpoint_path = resolve_checkpoint_path(checkpoint_path)
base_model_path = get_base_model(checkpoint_path)
vlm_cfg.model_id = base_model_path

system_prompt_path = PROJECT_ROOT.joinpath(Path("conf/prompts/objectnav_prompt.txt"))
with open(system_prompt_path,'r') as f:
    system_prompt = f.read()
rollout_cfg.convo_start_template=[
        {"role": "user", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ]

worker = RLWorker(asdict(rollout_cfg),**asdict(vlm_cfg))
worker._setup_peft(training_cfg)
worker.load_checkpoint(checkpoint_path,False,False)

def infer_step(worker,rgb_pil,instr_or_goal=None):
    '''
    rgb_pil: PIL RGB Image. 
    instr_or_goal: optional string
    '''
    if instr_or_goal is not None:
        worker.reset()
        worker.messages = substitute_convo_template(worker.rollout_config['convo_start_template'],{"instr_or_goal":instr_or_goal} | worker.rollout_config)
    action_probs,action_logprobs,outputs = worker.infer_probs(images=[rgb_pil],messages=worker.messages,temperature = worker.rollout_config['temperature'],pos_id_kwargs=None)
    action_id = np.random.choice(len(action_probs),p=action_probs) # sampling
    if action_id ==0 and worker.rollout_config['stop_prob_threshold'] is not None:
        if action_probs[0] >= worker.rollout_config['stop_prob_threshold']:
            action_id = 0
        else:
            action_id = np.random.choice(len(action_probs)-1,p=action_probs[1:]/np.sum(action_probs[1:]))+1
    worker.messages = substitute_convo_template(worker.rollout_config['convo_turn_template'],{"action":worker.rollout_config['action_space'][action_id]})
    return action_id,action_probs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block runs when the server starts
    print("Server starting... Worker is ready.")
    yield
    # This block runs when the server shuts down
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

@app.post("/act")
def get_action(
    file: UploadFile = File(...), 
    instr_or_goal: Optional[str] = Form(None)
):
    """
    Endpoint for the robot.
    - file: The camera image (binary).
    - instr_or_goal: Optional text. If provided, resets state.
    """
    # 1. Read Image
    # converting bytes directly to PIL is faster than saving to disk
    image_bytes = file.file.read()
    rgb_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Inference (Stateful)
    # We access the global 'worker' instance directly
    action_id, action_probs = infer_step(worker, rgb_pil, instr_or_goal)

    # 3. Return JSON
    return {
        "action_id": int(action_id),
        "action_probs": action_probs.tolist() # Convert numpy to list for JSON
    }

if __name__ == "__main__":
    # Run on 0.0.0.0 to expose to the internet (Vast AI requirements)
    # Adjust port as provided by Vast AI (usually passed via env vars or fixed)
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ssh -p 34929 -L 8000:localhost:8000 root@136.59.129.136 -N[]