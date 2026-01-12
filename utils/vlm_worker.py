# import os
import numpy as np
import torch
import time
import gc
import copy
import os
from typing import Optional, Any, List
from collections import defaultdict
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from dataclasses import dataclass,field
# from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half

class VLMWorker:
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct",attn_impl='sdpa',dtype='float16', prefix = '<|im_start|>assistant\n**',postfix = '**<|im_end|>',vocab=["stop","forward","left","right","up","down"],save_outputs=False,load_model=True,offload_cache=False,use_sparse=False,bev_canvas_size=2000):
        import transformers.modeling_flash_attention_utils as fa_utils
        def patched(position_ids, batch_size):
            return False
        fa_utils._is_packed_sequence = patched
        import torch
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.vocab = vocab
        self.vocab_ids = self._vocab_to_ids(vocab)
        self.save_outputs = save_outputs
        self.model_id = model_id
        self.attn_implementation = attn_impl
        self.dtype = dtype
        self.model=None
        self.prefix_ids = self.processor.tokenizer.encode(prefix)
        self.postfix_ids = self.processor.tokenizer.encode(postfix)
        self.offload_cache = offload_cache
        self.use_sparse = use_sparse
        self.bev_canvas_size = bev_canvas_size
        # Warmup the CUDA allocator
        if load_model:
            self.load_model()
        torch.cuda.empty_cache()
        self.reset()
        
    def reset(self):
        from transformers import DynamicCache,StaticCache
        import torch
        # self.past_key_values=StaticCache(config=self.model.config, offloading=self.offload_cache,max_cache_len=70000)
        self.past_key_values=None#DynamicCache(config=self.model.config, offloading=self.offload_cache)
        self.outputs = defaultdict(list)
        self.cumulative_inputs = None
        self.seq_keep_mask = None
        self.vis_keep_masks = []
        self.past_image_embeds = None #per batch list of image embed tensors of the form N_patch by N_hidden
        torch.cuda.empty_cache()

    def load_model(self):
        if not self.use_sparse:
            from transformers import AutoModelForImageTextToText
            print(f"Loading {self.model_id}...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,#"sdpa",
                device_map="cuda",
            ).eval()
            self.device = self.model.device
        else:
            from transformers import AutoConfig
            from utils.modeling import Qwen3VLSparseForConditionalGeneration
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            print(f"Loading {self.model_id} with sparsifying patch...")
            self.model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
                self.model_id, 
                config=config,
                device_map="cuda",
                dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation = self.attn_implementation).eval()
            self.device = self.model.device
        self.vl_model = self.model.model
        self.language_model = self.vl_model.language_model

    def tokenize_inputs(self,messages,images):
                # Process ONLY this turn's data
        text = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        inputs = self.processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt"
        )
        return inputs
    
    def _get_sandwich_indices(self, input_ids):
        import torch
        """
        Locates the indices of the logits that predict the sandwiched tokens.
        
        Returns:
            logit_indices (torch.Tensor): Indices relative to 'input_ids' to pass to logits_to_keep.
            target_ids (torch.Tensor): The ground truth tokens to calculate logprobs for.
        """
        # 1. Convert to NumPy for fast, robust search
        seq = input_ids[0].cpu().numpy()
        
        # Helper: NumPy sliding window search
        def search_sequence_numpy(arr, sub):
            window_size = len(sub)
            if len(arr) < window_size:
                return [-1]
            # Create strided view for O(1) comparison
            shape = (arr.shape[0] - window_size + 1, window_size)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            
            # Find all matches
            matches = np.all(windows == sub, axis=1)
            indices = np.where(matches)[0]
            
            return indices
        # 2. Find Prefix End
        prefix_np = np.array(self.prefix_ids)
        prefix_starts = search_sequence_numpy(seq, prefix_np)
        prefix_start = prefix_starts[-1]
        if prefix_start == -1:
            return None, None
        prefix_end = prefix_start + len(prefix_np)

        # 3. Find Postfix Start (Search after prefix)
        postfix_np = np.array(self.postfix_ids)
        seq_suffix = seq[prefix_end:] 
        postfix_relative_start = search_sequence_numpy(seq_suffix, postfix_np)
        assert(len(postfix_relative_start)==1) #1 prefix 1 postfix!
        postfix_relative_start=postfix_relative_start[0]

        if postfix_relative_start == -1:
            return None, None
        postfix_start = prefix_end + postfix_relative_start

        # 4. Calculate Indices
        # Target tokens are at: input_ids[prefix_end : postfix_start]
        # The hidden state at index 'i' predicts the token at 'i+1'.
        # So we need hidden states at: [prefix_end - 1 : postfix_start - 1]
        
        logit_start = prefix_end - 1
        logit_end = postfix_start - 1

        # Create the indices tensor to pass to the model
        logit_indices = torch.arange(logit_start, logit_end, device='cpu', dtype=torch.long)
        return logit_indices, prefix_starts, search_sequence_numpy(seq, postfix_np)

    def _vocab_to_ids(self,vocab):
        ids = []
        for word in vocab:
            ids +=self.processor.tokenizer.encode(word)
        if len(ids)!=len(vocab):
            raise("input vocabulary is not valid token list!")
        return ids
    
    # requires full inputs to work.
    def _accumulate_inputs(self,inputs):
        if self.cumulative_inputs is None:
            self.cumulative_inputs = dict(inputs.to('cpu'))
        else:
            self.cumulative_inputs['attention_mask'] = torch.cat([self.cumulative_inputs['attention_mask'],inputs['attention_mask']],dim=-1)
            # self.cumulative_inputs['position_ids'] = torch.cat([self.cumulative_inputs['position_ids'],inputs['position_ids']],dim=-1)
            self.cumulative_inputs['input_ids'] = torch.cat([self.cumulative_inputs['input_ids'],inputs['input_ids']],dim=-1)
            self.cumulative_inputs['image_grid_thw'] = torch.cat([self.cumulative_inputs['image_grid_thw'],inputs['image_grid_thw']],dim=0) # N_image by Hidden Size (16*16*6 ?)

    def _accumulate_custom_inputs(self,inputs,dim=0):
        if self.cumulative_inputs is None:
            self.cumulative_inputs = dict(inputs.to('cpu'))
        else:
            for k,v in inputs.items():
                if k in self.cumulative_inputs.keys():
                    self.cumulative_inputs[k] = torch.cat([self.cumulative_inputs[k],v],dim=dim)
                else:
                    self.cumulative_inputs[k] = v

    def render_cumulative_inputs(self,summarize_images = True):
        if not summarize_images:
            return self.processor.batch_decode(self.cumulative_inputs['input_ids'])
        else:
            # image_mask = self.cumulative_inputs['input_ids'] == self.processor.image_token_id
            sequences = [torch.unique_consecutive(sequence,return_counts=True) for sequence in self.cumulative_inputs['input_ids'].numpy()]
            return self.processor.batch_decode(sequences)
    
    def _calculate_pos_id(self,pos_id_kwargs=None):
        if pos_id_kwargs is None or pos_id_kwargs['mode'] == "standard":
            input_ids = self.cumulative_inputs['input_ids']
            image_grid_thw = self.cumulative_inputs['image_grid_thw']
            attention_mask = self.cumulative_inputs['attention_mask']
            # 1. Ask Qwen to calculate the 3D layout for this chunk
            # This returns positions starting at T=0, H=0, W=0 relative to this chunk
            position_ids, deltas = self.vl_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw, 
                video_grid_thw=None,
                attention_mask=attention_mask
            )
        elif pos_id_kwargs['mode'] == 'bev':
            from utils.bev_utils import get_pos_id
            self._accumulate_custom_inputs({'patch_coords':torch.tensor(pos_id_kwargs['patch_coords']).unsqueeze(0)},dim=0) 
            patch_coords = self.cumulative_inputs['patch_coords'] # N_image by H by W by 3
            patch_coords = patch_coords-torch.amin(patch_coords[:1],dim=[1,2],keepdim=True) 
            w,t,h = patch_coords[...,0],patch_coords[...,1],patch_coords[...,2] # horrific mess here
            w = w.reshape(-1)
            t = t.reshape(-1)
            h = h.reshape(-1)

            patch_coords = torch.stack([t,t+h,t+w],dim=0).reshape(1,3,-1)
            patch_coords = self.bev_canvas_size//2*torch.ones(1,3,1)
            position_ids = get_pos_id(self.cumulative_inputs['input_ids'],patch_coords.to(self.cumulative_inputs['input_ids'].dtype),self.processor,self.bev_canvas_size)
        return position_ids

    def _store_outputs(self, outputs):
        """
        Extracts cached tensors from ModelOutput and appends them to the rollout buffer.
        """
        # 1. Standard Tensors (Append to list, concatenate later)
        # These are already on CPU thanks to the TextModel code
        self.outputs['input_embeds'].append(outputs.input_embeds)
        self.outputs['position_ids'].append(outputs.position_ids)
        self.outputs['visual_pos_masks'].append(outputs.visual_pos_masks)
        
        # 2. Deepstack Inputs (List of Tensors handling)
        # outputs.deepstack_inputs is a list [Layer1_Tensor, Layer2_Tensor, ...]
        # We need to store them so we can eventually concat Layer 1 across all time steps.
        if outputs.deepstack_inputs is not None:
            if 'deepstack_inputs' not in self.outputs:
                # Initialize list of lists: [[], [], [], ...]
                self.outputs['deepstack_inputs'] = [[] for _ in outputs.deepstack_inputs]
            
            # Append Layer K's tensor to the Kth list
            for layer_idx, layer_tensor in enumerate(outputs.deepstack_inputs):
                self.outputs['deepstack_inputs'][layer_idx].append(layer_tensor)

    def infer_step(self,messages,images,full_logprobs=False,temperature=1.2,check_probs=True,crop_inputs=True,pos_id_kwargs=None):
        if self.model is None:
            self.load_model()
            self.reset()
        turn_inputs = self.tokenize_inputs(messages,images)
        # First we must crop the sequence so the turns properly lign up.
        logit_indices,prefix_starts,postfix_starts = self._get_sandwich_indices(turn_inputs['input_ids'])
        if crop_inputs:
            if len(prefix_starts)>1:
                turn_inputs['attention_mask'] = turn_inputs['attention_mask'][:,(postfix_starts[0]-1):(postfix_starts[-1]-1)]
                turn_inputs["input_ids"] = turn_inputs['input_ids'][:,(postfix_starts[0]-1):(postfix_starts[-1]-1)]
            else:
                turn_inputs['attention_mask'] = turn_inputs['attention_mask'][:,:(postfix_starts[-1]-1)]
                turn_inputs["input_ids"] = turn_inputs['input_ids'][:,:(postfix_starts[-1]-1)]
        self._accumulate_inputs(turn_inputs)
        turn_inputs = {k: v.to(self.device) for k, v in turn_inputs.items()}
        self.cumulative_inputs['position_ids'] = self._calculate_pos_id(pos_id_kwargs) # calculate the pos_ids for the whole sequence. hopefully not too expensive...
        current_len = turn_inputs['input_ids'].shape[1]
        turn_inputs['position_ids'] = self.cumulative_inputs['position_ids'][..., -current_len:].to(self.device)
        if self.use_sparse:
            turn_inputs['past_image_embeds'] = self.past_image_embeds
            turn_inputs['save_image_db'] = True # new argument in sparse qwen to signal keeping the db as internal state
            # sparsify the input attention mask
            turn_inputs['attention_mask'] = None#turn_inputs['attention_mask'] = torch.ones((turn_inputs['input_ids'].shape[0], (self.past_key_values.get_seq_length() if self.past_key_values is not None else 0) + turn_inputs['input_ids'].shape[1]), device=self.device, dtype=turn_inputs['attention_mask'].dtype)# torch.ones(1,seql,device=self.device)
        else:
            turn_inputs['attention_mask'] = self.cumulative_inputs['attention_mask'].to(self.device)
        if self.save_outputs:
            turn_inputs['save_embeds'] = True

        with torch.inference_mode():
            # t0 = time.time()
            outputs = self.model.forward(
                **turn_inputs,
                past_key_values=self.past_key_values,
                use_cache=True,
                # logits_to_keep = logit_indices.to(self.model.device)
                logits_to_keep=1
            )
            # print(f"latency: {time.time()-t0}")
            self.past_key_values = outputs['past_key_values']
            if self.save_outputs:
                self._store_outputs(outputs)
             # Compute logprobs directly (1-to-1 mapping)
            relevant_logits = outputs.logits[0].float()
            all_logprobs = torch.log_softmax(relevant_logits/temperature, dim=-1)
            if self.use_sparse:
                current_keep_mask = self.language_model.seq_keep_mask
                self.vis_keep_masks.append(self.language_model.vis_keep_mask.cpu())
                if self.seq_keep_mask is None:
                    self.seq_keep_mask = current_keep_mask.cpu()
                else:
                    self.seq_keep_mask = torch.cat((self.seq_keep_mask,current_keep_mask.cpu()))
                if self.past_image_embeds is None:
                    self.past_image_embeds = self.language_model.kept_visual_embeds
                else:
                    for idx, image_embeds in enumerate(self.language_model.kept_visual_embeds):
                        self.past_image_embeds[idx] = torch.cat((self.past_image_embeds[idx],image_embeds)) #handle the batching...
        if check_probs:
            try:
                assert(torch.argmax(all_logprobs,dim=-1).item() in self.vocab_ids)
            except:
                print("WARNING: prediction not in provided vocab")
        # print("inference done!")
        if full_logprobs:
            return all_logprobs.cpu().float().numpy()
        else:
            return all_logprobs[:,self.vocab_ids].cpu().float().numpy()

    def infer_probs(self,messages,images,**kwargs):
        logprobs = self.infer_step(messages,images,**kwargs)
        assert(len(logprobs)==1) #ensure there is a unique token position for decision making
        logprobs = logprobs[0]
        probs = np.exp(logprobs)
        probs /= np.sum(probs)
        return probs


@dataclass
class VLMTrainingConfig:
    # Optimization
    learning_rate: float = 5e-6
    grad_accum_steps: int = 1
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True

    # Value Head Configuration
    value_head_learning_rate: float = 5e-4  # Often higher than Adapter LR
    value_head_dropout: float = 0.1
    # List of hidden layer sizes. Empty list [] implies a single linear layer (Linear Probe).
    value_head_hidden_dims: List[int] = field(default_factory=list)
    # PEFT: Pass the actual configuration object here (e.g., LoraConfig)
    # Typed as Any to avoid crashing if peft isn't installed on the driver
    peft_config: Optional[Any] = None
    


# Handle optional PEFT imports gracefully
try:
    from peft import get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class ValueHead(nn.Module):
    """
    A configurable MLP Value Head.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.Mish()) # why not
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        # Final projection to scalar value
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

@dataclass
class RLBatch:
    """
    Represents a batch of full sequences (B, T) ready for training.
    """
    # --- Static Data (From Rollout) ---
    input_ids: torch.Tensor          # (B, T) - For logging/masks
    inputs_embeds: torch.Tensor      # (B, T, D) - Cached from Worker (Heavy)
    attention_mask: torch.Tensor     # (B, T)
    response_mask: torch.Tensor      # (B, T) - 1 for generated tokens, 0 for prompt
    
    actions: torch.Tensor            # (B, T)
    old_log_probs: torch.Tensor      # (B, T) - π_old
    rewards: torch.Tensor            # (B, T) - Dense rewards at every step
    
    # --- Computed Data (Filled by Driver) ---
    values: Optional[torch.Tensor] = None     # (B, T) - V(s)
    advantages: Optional[torch.Tensor] = None # (B, T) - A(s,a)
    returns: Optional[torch.Tensor] = None    # (B, T) - Target for Value Head

class VLMTrainingMixin:
    def setup_training(self, config: VLMTrainingConfig,rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,):
        """
        Sets up distributed training using the provided TrainConfig.
        """
        from accelerate import DistributedDataParallelKwargs

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #prevent value head from being killed
        # 1. Manual Environment Injection for Ray
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        
        # 2. Initialize Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.grad_accum_steps,
            mixed_precision=config.mixed_precision,
            kwargs_handlers=[kwargs]
        )

        # 3. Gradient Checkpointing (Must run before PEFT wrapping)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
            # This logic handles the edge case where input embeddings are frozen
            # causing backward() to fail with checkpointing enabled.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads() # use_reentrant=False to prevent hangs
            else:
                raise NotImplementedError("Model does not support 'enable_input_require_grads' method.")
        hidden_size = self.model.language_model.config.hidden_size
        
        self.model.value_head = ValueHead(
            input_dim=hidden_size,
            hidden_dims=config.value_head_hidden_dims,
            dropout=config.value_head_dropout
        )
        # Ensure value head is in the same dtype as the model base
        self.model.value_head.to(self.model.dtype)
        # 4. Apply PEFT (if config provided)
        if config.peft_config is not None:
            if not PEFT_AVAILABLE:
                raise ImportError("TrainConfig has peft_config, but 'peft' library is not installed.")
            # Direct application of the config object
            # CRITICAL: Add 'value_head' to modules_to_save so PEFT treats it as 
            # a full-rank trainable module (not an adapter) and saves it in the checkpoint.
            if config.peft_config.modules_to_save is None:
                config.peft_config.modules_to_save = []
            if "value_head" not in config.peft_config.modules_to_save:
                print("saving value head")
                config.peft_config.modules_to_save.append("value_head")
            self.model = get_peft_model(self.model, config.peft_config)
            # Print trainable parameters to verify LoRA is active
            if self.accelerator.is_local_main_process:
                self.model.print_trainable_parameters()
        # 5. Create Optimizer
        # Only optimize parameters that require gradients (i.e., the Adapters)

        head_params = [p for n, p in self.model.named_parameters() if "value_head" in n and p.requires_grad]
        rest_params = [p for n, p in self.model.named_parameters() if "value_head" not in n and p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": head_params,
                "lr": config.value_head_learning_rate,
                "name": "value_head"
            },
            {
                "params": rest_params,
                "lr": config.learning_rate,
                "name": "adapters"
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

        # 6. Prepare with Accelerator
        # self.ddp_model becomes the sync-wrapper
        # self.model remains the direct reference (now with LoRA layers attached)
        self.ddp_model, self.optimizer = self.accelerator.prepare(
            self.model, optimizer
        )
        
    def train_sft_step(self, batch):
        """
        Standard training step.
        """
        self.ddp_model.train()
        
        # Accumulate gradients (handle micro-batches)
        with self.accelerator.accumulate(self.ddp_model):
            # Forward via DDP wrapper (triggers sync)
            outputs = self.ddp_model(**batch)
            loss = outputs.loss
            
            # Backward (handles mixed precision scaling)
            self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def save_adapter(self, path):
        """
        Saves ONLY the LoRA adapters. 
        Safe to call from Ray actor (handles rank check internally).
        """
        # Wait for all workers to finish their current step
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            # We unwrap to get the PeftModel, then call save_pretrained
            # which knows to only save the 'adapter_model.bin'
            unwrapped = self.accelerator.unwrap_model(self.ddp_model)
            unwrapped.save_pretrained(path)
            print(f"Adapters saved to {path}")

class DataGenerator:
    """Generates synthetic turn data."""
    def __init__(self, width=640, height=480, processor=None):
        self.width = width
        self.height = height
        self.processor = processor

    def create_synthetic_image(self):
        from PIL import Image
        arr = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def _prepare_turn_inputs(self, step_idx):
        """
        Creates inputs for a SINGLE turn (Image + Text).
        We do not build the full conversation history in the prompt.
        We rely on the KV cache for history.
        """
        image = self.create_synthetic_image()
        
        # Construct a standalone prompt for this step
        # We simulate the user asking for a move
        messages = [

            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Step {step_idx}: Next move?"}
                ]
            },
            # We add the Assistant start token to force the model to predict the response immediately
            {"role": "assistant", "content": "**forward**"} 
        ]
        return messages, [image]
 
if __name__ == "__main__":
    from vlm_worker import VLMWorker
    import torch
    import time
    import argparse
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print("running inference test")
    # --- Constants ---
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    # MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480

    
    worker = VLMWorker(model_id=MODEL_ID,attn_impl='flash_attention_2', dtype='bfloat16',offload_cache=False,use_sparse=True)
    generator = DataGenerator(IMAGE_WIDTH, IMAGE_HEIGHT, worker.processor)
    worker.reset()
    from tqdm import tqdm
    # torch.cuda.memory._record_memory_history(
    #    max_entries=3
    # )
    for i in tqdm(range(160)):
        messages,images = generator._prepare_turn_inputs(i)
        action = worker.infer_probs(messages,images)
        # action = worker.infer_step(messages,images)
