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
from torch.optim import AdamW
from dataclasses import dataclass,field
# from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
import torch.nn.functional as F

def compute_full_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Computes the token-level KL divergence: KL(pi || ref) = sum(pi * (log_pi - log_ref))
    
    Args:
        log_probs: [Batch, Seq, Vocab] (Normalized, i.e., LogSoftmax applied)
        ref_log_probs: [Batch, Seq, Vocab] (Normalized, i.e., LogSoftmax applied)
    
    Returns:
        kl_penalty: [Batch, Seq] (Scalar KL value per token)
    """
    # 1. Convert log_probs to probs for the weighting term
    probs = log_probs.exp()
    
    # 2. Compute KL: P * (log_P - log_Q)
    #    We sum over the last dimension (Vocab/Action Space)
    kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
    
    return kl

class VLMWorker:
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct",attn_impl='sdpa',dtype='float16', prefix = '<|im_start|>assistant\n**',postfix = '**<|im_end|>',vocab=["stop","forward","left","right","up","down"],save_outputs=False,load_model=True,offload_cache=False,use_sparse=False,bev_canvas_size=2000,save_pixels=False,sparse_thresh="error"):
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
        self.save_pixels = save_pixels
        self.model_id = model_id
        self.attn_implementation = attn_impl
        self.dtype = dtype
        self.model=None
        self.prefix_ids = self.processor.tokenizer.encode(prefix)
        self.postfix_ids = self.processor.tokenizer.encode(postfix)
        self.offload_cache = offload_cache
        self.use_sparse = use_sparse
        self.bev_canvas_size = bev_canvas_size
        self.sparse_thresh = sparse_thresh
        self._is_merged = None
        self._is_lora = None
        # Warmup the CUDA allocator
        if load_model:
            self.load_model()
        torch.cuda.empty_cache()
        self.reset()
        
    def reset(self):
        from transformers import DynamicCache,StaticCache
        import torch
        self.offset=0
        # self.past_key_values=StaticCache(config=self.model.config, offloading=self.offload_cache,max_cache_len=70000)
        self.past_key_values=None#DynamicCache(config=self.model.config, offloading=self.offload_cache)
        self.outputs = defaultdict(list)
        self.cumulative_inputs = None
        self.seq_keep_mask = None
        self.vis_keep_masks = []
        self.past_image_embeds = None #per batch list of image embed tensors of the form N_patch by N_hidden
        self.logit_indices = []
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
        self.model.config.use_cache = False
        self.model.to('cuda')
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
        # print(prefix_np)
        # print()
        # print(seq)
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
            if self.save_pixels:
                self.cumulative_inputs['pixel_values'] = [inputs['pixel_values'].to('cpu')]
        else:
            self.cumulative_inputs['attention_mask'] = torch.cat([self.cumulative_inputs['attention_mask'],inputs['attention_mask']],dim=-1)
            # self.cumulative_inputs['position_ids'] = torch.cat([self.cumulative_inputs['position_ids'],inputs['position_ids']],dim=-1)
            self.cumulative_inputs['input_ids'] = torch.cat([self.cumulative_inputs['input_ids'],inputs['input_ids']],dim=-1)
            self.cumulative_inputs['image_grid_thw'] = torch.cat([self.cumulative_inputs['image_grid_thw'],inputs['image_grid_thw']],dim=0) # N_image by Hidden Size (16*16*6 ?)
            if self.save_pixels:
                self.cumulative_inputs['pixel_values'].append(inputs['pixel_values'].to('cpu'))
    
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
    
    def _pos_id_fast(self,turn_inputs):
        # fast version that only calculates pos ids for the current turn.
        input_ids = turn_inputs['input_ids']
        image_grid_thw = turn_inputs['image_grid_thw']
        attention_mask = turn_inputs['attention_mask']
        position_ids, deltas = self.vl_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw, 
            video_grid_thw=None,
            attention_mask=attention_mask
        )
        position_ids += self.offset
        self.offset += len(turn_inputs['input_ids'][0])
        self.offset += deltas.item()
        return position_ids

    def _store_outputs(self, outputs):
        """
        Extracts cached tensors from ModelOutput and appends them to the rollout buffer.
        """
        # 1. Standard Tensors (Append to list, concatenate later)
        # These are already on CPU thanks to the TextModel code
        self.outputs['inputs_embeds'].append(outputs.inputs_embeds)
        self.outputs['position_ids'].append(outputs.position_ids)
        self.outputs['visual_pos_masks'].append(outputs.visual_pos_masks)
        
        # 2. Deepstack Inputs (List of Tensors handling)
        # outputs.deepstack_visual_embeds is a list [Layer1_Tensor, Layer2_Tensor, ...]
        # We need to store them so we can eventually concat Layer 1 across all time steps.
        if outputs.deepstack_visual_embeds is not None:
            if 'deepstack_visual_embeds' not in self.outputs:
                # Initialize list of lists: [[], [], [], ...]
                self.outputs['deepstack_visual_embeds'] = [[] for _ in outputs.deepstack_visual_embeds]
            
            # Append Layer K's tensor to the Kth list
            for layer_idx, layer_tensor in enumerate(outputs.deepstack_visual_embeds):
                self.outputs['deepstack_visual_embeds'][layer_idx].append(layer_tensor)
    def _get_sparse_logit_indices(self):
        ranks = self.seq_keep_mask.long().cumsum(dim=0)
        logits_to_keep = ranks[self.logit_indices] - 1 # logit indices of the sparsified sequence
        return logits_to_keep
    
    def _pack_embeds(self):
        '''
        pack all the embeds needed to replicate forward pass of the entire sequence.
        
        RESETS internal outputs after packing.
        '''
        deepstack =[torch.cat([self.outputs['deepstack_visual_embeds'][i][j] for j in range(len(self.outputs['deepstack_visual_embeds'][i]))],dim=0) for i in range(len(self.outputs['deepstack_visual_embeds']))]
        position_ids = torch.cat(self.outputs['position_ids'],dim=-1)
        visual_pos_masks = torch.cat(self.outputs['visual_pos_masks'],dim=1)
        inputs_embeds = torch.cat(self.outputs['inputs_embeds'],dim=1)
        input_ids = self.cumulative_inputs['input_ids'][:,self.seq_keep_mask]
        self.outputs = defaultdict(list) # reset outputs.
        
        
        return {
            "deepstack_visual_embeds": torch.stack(deepstack,dim=0).cpu(), #N_layer by N_patch by N_hidden
            "position_ids": position_ids.cpu(),
            "visual_pos_masks": visual_pos_masks.cpu(),
            "inputs_embeds": inputs_embeds.cpu(),
            "input_ids_reference": input_ids.cpu(),
            "logits_to_keep": self._get_sparse_logit_indices().cpu()  
        }

    def _pack_inputs(self):
        '''
        pack all the raw inputs needed to replicate forward pass of the entire sequence.
        "logits_to_keep" ensure only action tokens are used by the lmhead.
        '''
        input_ids = self.cumulative_inputs['input_ids']
        attention_mask = self.cumulative_inputs['attention_mask']
        image_grid_thw = self.cumulative_inputs['image_grid_thw']
        pixel_values = self.cumulative_inputs.get('pixel_values',None)
        if pixel_values is not None and isinstance(pixel_values,list):
            pixel_values = torch.cat(pixel_values,dim=0)
        position_ids = self._calculate_pos_id()
        self.cumulative_inputs = None # reset cumulative inputs.
        return {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "image_grid_thw": image_grid_thw.cpu(),
            "position_ids": position_ids.cpu(),
            "pixel_values": pixel_values.cpu() if pixel_values is not None else None,
            "seq_keep_mask": self.seq_keep_mask.cpu(),
            "vis_keep_mask": torch.cat(self.vis_keep_masks,dim=0).cpu(),
            "logits_to_keep": self._get_sparse_logit_indices().cpu()
        }

    def infer_step(self,messages,images,full_logprobs=False,temperature=1.0,check_probs=True,crop_inputs=True,pos_id_kwargs=None):
        t0 = time.time()
        self.model.gradient_checkpointing_disable()
        self.model.eval()

        if self.model is None:
            self.load_model()
            self.reset()
        if self.using_lora() and not self.is_merged():
            self.merge_adapter() # for inference speed
            pass
        # print(f"lora merge time: {time.time()-t0}",end=" ")
        
        t = time.time()
        turn_inputs = self.tokenize_inputs(messages,images)
        # print(f"tokenize time: {time.time()-t}",end=" ")
        # First we must crop the sequence so the turns properly lign up.
        logit_indices,prefix_starts,postfix_starts = self._get_sandwich_indices(turn_inputs['input_ids'])
        if crop_inputs:
            if len(prefix_starts)>1:
                turn_inputs['attention_mask'] = turn_inputs['attention_mask'][:,(postfix_starts[0]-1):(postfix_starts[-1]-1)]
                turn_inputs["input_ids"] = turn_inputs['input_ids'][:,(postfix_starts[0]-1):(postfix_starts[-1]-1)]
            else:
                turn_inputs['attention_mask'] = turn_inputs['attention_mask'][:,:(postfix_starts[-1]-1)]
                turn_inputs["input_ids"] = turn_inputs['input_ids'][:,:(postfix_starts[-1]-1)]
        
        t = time.time()
        self._accumulate_inputs(turn_inputs)
        # print(f"accumulate time: {time.time()-t}",end=" ")
        self.logit_indices.append(self.cumulative_inputs['input_ids'].shape[1]-1) #slice index for the hidden state predicting the last token in this turn.
        turn_inputs = {k: v.to(self.device) for k, v in turn_inputs.items()}
        t = time.time()
        if pos_id_kwargs is None or pos_id_kwargs['mode'] == "standard": # use fast pos id calculation
            turn_inputs['position_ids'] = self._pos_id_fast(turn_inputs)
            if 'position_ids' not in self.cumulative_inputs.keys():
                self.cumulative_inputs['position_ids'] = turn_inputs['position_ids'].to('cpu')
            else:
                self.cumulative_inputs['position_ids'] = torch.cat([self.cumulative_inputs['position_ids'],turn_inputs['position_ids'].to('cpu')],dim=-1)
        else:
            self.cumulative_inputs['position_ids'] = self._calculate_pos_id(pos_id_kwargs) # calculate the pos_ids for the whole sequence. hopefully not too expensive...
            turn_inputs['position_ids'] = self.cumulative_inputs['position_ids'][..., -current_len:].to(self.device)
        # print(f"pos id time: {time.time()-t}",end=" ")
         # Set up inputs for this turn
        current_len = turn_inputs['input_ids'].shape[1]
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
            t = time.time()
            outputs = self.model.forward(
                **turn_inputs,
                past_key_values=self.past_key_values,
                use_cache=True,
                # logits_to_keep = logit_indices.to(self.model.device)
                logits_to_keep=1,
                sparse_thresh = self.sparse_thresh
            )
            self.past_key_values = outputs['past_key_values']
             # Compute logprobs directly (1-to-1 mapping)
            relevant_logits = outputs.logits[0].float()
            if not full_logprobs:
                relevant_logits = relevant_logits[...,self.vocab_ids]
            if np.abs(temperature-1.0) > 1e-7:
                logprobs = torch.log_softmax(relevant_logits/temperature, dim=-1)
            else:
                logprobs = torch.log_softmax(relevant_logits, dim=-1)
            # print(f"vlm latency: {time.time()-t}",end=" ")

            if self.save_outputs:
                t = time.time()
                self._store_outputs(outputs)
                # print(f"store outputs time: {time.time()-t}",end=" ")
            if self.use_sparse:
                t = time.time()
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
                # print(f"store sparse states time: {time.time()-t}",end=" ")
        # if check_probs:
        #     try:
        #         assert(torch.argmax(logprobs,dim=-1).item() in self.vocab_ids)
        #     except:
        #         print("WARNING: prediction not in provided vocab")
        # # print("inference done!")
        # print(f" total time: {time.time()-t0}")
        return logprobs.cpu().float().numpy(),outputs

        
    def _calculate_action_logprobs(self,logits):
        import torch
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits)
        action_logprobs = torch.log_softmax(logits[...,self.vocab_ids],dim=-1)
        return action_logprobs
    
    def infer_probs(self,messages,images,**kwargs):
        logprobs,outputs = self.infer_step(messages,images,**kwargs)
        assert(len(logprobs)==1) #ensure there is a unique token position for decision making
        logprobs = logprobs[0]
        probs = np.exp(logprobs)
        probs /= np.sum(probs)
        return probs,logprobs,outputs

    def merge_adapter(self):
        print("Merging LoRA adapters for inference...")
        self._is_merged = True
        self.model.merge_adapter()
        
    def unmerge_adapter(self):
        print("Unmerging LoRA for training")
        self._is_merged = False
        self.model.unmerge_adapter()
        
    def is_merged(self):
        if self._is_merged is None:
            self._is_merged = len(self.model.get_model_status().merged_adapters) > 0
        return self._is_merged
    
    def using_lora(self):
        if self._is_lora is None:
            self._is_lora = isinstance(self.model,PeftModel)
        return self._is_lora

# Handle optional PEFT imports gracefully
try:
    from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class ValueHead(nn.Module):
    """
    A configurable MLP Value Head.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1,dtype:str='float32'):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim,dtype=dtype))
            layers.append(nn.Mish()) # why not
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        # Final projection to scalar value
        final_proj = nn.Linear(curr_dim, 1,dtype=dtype)
        # Initialize to zero for 0 value at start
        # with torch.no_grad():
        #     final_proj.weight.fill_(0.)
        #     final_proj.bias.fill_(0.)
        layers.append(final_proj)
        self.mlp = nn.Sequential(*layers)
        self.dtype = dtype

    def forward(self, x):
        return self.mlp(x)
    
class VLMWrapper(nn.Module):
    """
    Thin wrapper that enables forward pass of the language model to play nicely with DDP
    """
    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm # Can be PeftModel
        self._freeze_vision_tower()

    def _forward_embeds(self,embeds_inputs,compute_values=False,value_grad_scale=0.1):
        embeds_inputs = {k:v.to('cuda') for k,v in embeds_inputs.items()}
        embeds_inputs['inputs_embeds'] = embeds_inputs['inputs_embeds'].to(self.vlm.dtype)
        if self.vlm.training:
            embeds_inputs['inputs_embeds'].requires_grad_(True)
        embeds_inputs['deepstack_visual_embeds'] = [v.to(self.vlm.dtype) for v in embeds_inputs['deepstack_visual_embeds']]
        logits_to_keep = embeds_inputs.pop('logits_to_keep')
        embeds_inputs.pop('input_ids_reference')
        embeds_inputs['seq_keep_mask']='everything' # force keeping everything since seq is already sparse
        hidden = self.vlm.language_model(**embeds_inputs).last_hidden_state
        values = None
        if compute_values:
            value_hidden = hidden[:,logits_to_keep].to(self.vlm.value_head.dtype)
            if value_grad_scale<=0:
                # 1. Fully Detached (Old way)
                value_hidden = value_hidden.detach()
            
            else:             
                # Forward: Identity. Backward: Gradient * scale.
                value_hidden = (value_hidden * value_grad_scale) + (value_hidden.detach() * (1 - value_grad_scale))


            values = self.vlm.value_head(value_hidden).squeeze(-1)
        logits = self.vlm.lm_head(hidden[:,logits_to_keep])
        return logits,values

    def forward(self, mode = "embeds_inputs",**inputs):
        if mode == "embeds_inputs":
            return self._forward_embeds(**inputs)
        elif mode == "standard":
            if hasattr(self.vlm, "value_head"):
                # Calculate a 0.0 scalar attached to the value head's graph
                dummy_loss = 0.0 # this hack prevents ddp freeze in sft
                for p in self.vlm.value_head.parameters():
                    if p.requires_grad:
                        dummy_loss = dummy_loss + p.sum() * 0.0
                        break
            return self.vlm(**inputs)
        elif mode == "language":
            return self.vlm.language_model(**inputs)

    def _freeze_vision_tower(self):
        """
        Locates the vision tower and ensures all parameters are frozen.
        Logs a warning if trainable parameters were found and suppressed.
        """
        # 1. unwrapping helper to get down to the base architecture
        # (Handles PeftModel, DistributedDataParallel, etc.)
        base = self.vlm

        # 2. Attempt to locate the vision module using common naming conventions
        # (Covers LLaVA, Qwen-VL, Idefics, etc.)
        vision_tower = None
        potential_names = ["vision_model", "vision_tower", "visual_model", "visual", "vit"]
        
        # Check top level
        for attr in potential_names:
            if hasattr(base, attr):
                vision_tower = getattr(base, attr)
                break
        
        # Check inside .model (Common in HF Llama-based architectures)
        if vision_tower is None and hasattr(base, "model"):
             for attr in potential_names:
                if hasattr(base.model, attr):
                    vision_tower = getattr(base.model, attr)
                    break

        # 3. Freeze and Warn
        if vision_tower is not None:
            frozen_count = 0
            example_names = []
            
            for name, param in vision_tower.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
                    if len(example_names) < 3:
                        example_names.append(name)
            
            if frozen_count > 0:
                print(f"\n[VLMWrapper] ⚠️ WARNING: Found {frozen_count} trainable parameters in Vision Tower.")
                print(f"[VLMWrapper] Examples: {example_names}")
                print("[VLMWrapper] ACTION: Forcibly FROZEN these parameters to ensure DDP compatibility in RL steps.\n")
        else:
            # Fallback info if architecture is exotic
            print("[VLMWrapper] Info: Could not auto-detect Vision Tower module to safeguard. Assuming it is correctly frozen.")

class VLMTrainingMixin:
    from config_schema import VLMTrainingConfig

    def setup_training(self, config: VLMTrainingConfig, rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,):
        """
        Sets up distributed training using the provided TrainConfig.
        """
        from accelerate import DistributedDataParallelKwargs
        from transformers import get_scheduler
        from accelerate import Accelerator

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #prevent value head from being killed
        # 1. Manual Environment Injection for Ray
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        
        self.rl_algo_config=config.rl_config
        # 2. Initialize Accelerator
        print("creating accelerator")
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.grad_accum_steps,
            mixed_precision=config.mixed_precision,
            kwargs_handlers=[kwargs]
        )
# or check the accelerator state
        self.gradient_checkpointing =config.gradient_checkpointing
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
        
        if config.rl_config is not None:
            if config.rl_config.use_value:
                self.model.value_head = ValueHead(
                    input_dim=hidden_size,
                    hidden_dims=config.value_head_hidden_dims,
                    dropout=config.value_head_dropout,
                    dtype=getattr(torch,config.value_head_dtype)
                ).to(self.model.device)
            from verl.trainer.ppo.core_algos import get_policy_loss_fn
            self.policy_loss_fn = get_policy_loss_fn(config.rl_config.policy_loss_name)
        # 4. Apply PEFT (if config provided)
        self._setup_peft(config)
            # Print trainable parameters to verify LoRA is active
        try:
            if self.accelerator.is_local_main_process:
                self.model.print_trainable_parameters()
        except:
            print("failed to print trainable parameters...")
        # 5. Create Optimizer
        # Only optimize parameters that require gradients (i.e., the Adapters)
        print(f"accelerator device: {self.accelerator.device}")
        wrapper = VLMWrapper(self.model)
        rest_params = [p for n, p in wrapper.named_parameters() if "value_head" not in n and p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": rest_params,
                "lr": config.learning_rate,
                "name": "adapters"
            }
        ]

        if config.rl_config.use_value:
            head_params = [p for n, p in wrapper.named_parameters() if "value_head" in n and p.requires_grad]
            optimizer_grouped_parameters+=[
                {
                    "params": head_params,
                    "lr": config.value_head_learning_rate,
                    "name": "value_head"
                }]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps, # Short warmup usually sufficient for RL
            num_training_steps=config.total_optimization_steps
        )

        # 6. Prepare with Accelerator
        # self.ddp_model becomes the sync-wrapper
        # self.model remains the direct reference (now with LoRA layers attached)

        self.ddp_model, self.optimizer,self.scheduler = self.accelerator.prepare(
            wrapper, optimizer, scheduler
        )            

        if config.checkpoint is not None:
            self.load_checkpoint(config.checkpoint,True,config.load_optim,config.load_sched)
    def _setup_peft(self, config):
        if config.peft_config is not None:
            if not PEFT_AVAILABLE:
                raise ImportError("TrainConfig has peft_config, but 'peft' library is not installed.")
            from peft import LoraConfig
            from dataclasses import asdict
            # Direct application of the config object
            # CRITICAL: Add 'value_head' to modules_to_save so PEFT treats it as 
            # a full-rank trainable module (not an adapter) and saves it in the checkpoint.
            if config.peft_config.modules_to_save is None:
                config.peft_config.modules_to_save = []
            if "value_head" not in config.peft_config.modules_to_save:
                config.peft_config.modules_to_save.append("value_head")
            try:
                peft_kwargs = asdict(config.peft_config)
            except:
                from omegaconf import OmegaConf
                peft_kwargs = OmegaConf.to_container(config.peft_config,resolve=True)
            for key in ["target_modules", "modules_to_save", "modules_to_freeze"]:
                if key in peft_kwargs and peft_kwargs[key] is not None:
                    # The magic fix: list() casts ListConfig -> list
                    peft_kwargs[key] = list(peft_kwargs[key])

            real_peft_config = LoraConfig(**peft_kwargs)
            self.model = get_peft_model(self.model, real_peft_config)
        else:
            print("PEFT config not provided; training all model parameters.")
    def train_sft_step(self, batch):
        """
        Standard training step.
        """
        self.ddp_model.train()
        if self.is_merged():
            self.unmerge_adapter()
        self.accelerator.wait_for_everyone() # ensure all workers have unmerged before training
        # Accumulate gradients (handle micro-batches)
        with self.accelerator.accumulate(self.ddp_model):
            # Forward via DDP wrapper (triggers sync)
            outputs = self.ddp_model(mode="standard",**batch)
            loss = outputs.loss
            
            # Backward (handles mixed precision scaling)
            self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def _forward_embeds(self,rl_embeds_inputs,model=None,compute_values=False):
        if model is None:
            model = self.model
        embeds_inputs = {k:v.to('cuda') for k,v in rl_embeds_inputs.items()}
        embeds_inputs['inputs_embeds'] = embeds_inputs['inputs_embeds'].to(self.model.dtype)
        if model.training:
            embeds_inputs['inputs_embeds'].requires_grad_(True)
        embeds_inputs['deepstack_visual_embeds'] = [v.to(self.model.dtype) for v in embeds_inputs['deepstack_visual_embeds']]
        logits_to_keep = embeds_inputs.pop('logits_to_keep')
        embeds_inputs.pop('input_ids_reference')
        embeds_inputs['seq_keep_mask']='everything' # force keeping everything since seq is already sparse
        hidden = model.language_model(**embeds_inputs,).last_hidden_state
        values = None
        if compute_values:
            values = model.value_head(hidden[:,logits_to_keep].to(model.value_head.dtype)).squeeze(-1)
        logits = model.lm_head(hidden[:,logits_to_keep])
        return logits,values
    
    def _forward_seq(self,rl_seq_inputs):
        # seq_inputs = {k:torch.tensor(v,device='cuda') for k,v in self.rl_seq_inputs.items()}
        seq_inputs = {k:v.to('cuda') for k,v in rl_seq_inputs.items()}
        output = self.model(**seq_inputs)
        return output.logits
    
    def _setup_training(self):
        self.ddp_model.train()
        if self.is_merged():
            self.unmerge_adapter()
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
        self.reset() #clear internal state, training is (mostly) stateless
        self.accelerator.wait_for_everyone() # ensure all workers have unmerged before training
        
    def _training_forward(self,embeds_inputs):
        # Forward via DDP wrapper (triggers sync)
        logits,vpreds = self.ddp_model(embeds_inputs = embeds_inputs,compute_values = self.rl_algo_config.use_value,value_grad_scale=self.rl_algo_config.value_grad_scale)
        return logits,vpreds 
    
    def rl_loss(self, log_probs, actions, advantages, response_mask, old_log_prob, returns, old_values, vpreds, logits, rollout_log_probs=None, ref_log_probs=None):        
        from verl.trainer.ppo.core_algos import compute_value_loss,compute_entropy_loss
        #TODO: rollout correction, rejection sampling to exclude bad tokens
        log_prob = torch.gather(log_probs, -1, actions.unsqueeze(-1).to(log_probs.device)).squeeze(-1)
        response_mask = response_mask.to(log_prob.device).bool()
        # --- CRITICAL FIX: Handle Pure DAgger Episodes ---
        if response_mask.sum() == 0:
            print("warning: empty RL mask, skipping RL loss.")
            # If PPO has no data (all tokens went to DAgger), return 0 loss safely.
            # We strictly require grad=True for DDP compatibility.
            zero_loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
            return zero_loss, {'loss/pg_loss': 0.0, 'return': 0.0, 'train/vf_loss': 0.0}
        pg_loss,metrics = self.policy_loss_fn(old_log_prob=old_log_prob.to(log_prob.device),log_prob=log_prob,advantages=advantages.to(log_prob.device),response_mask=response_mask,config = self.rl_algo_config)
        metrics['loss/pg_loss'] = pg_loss.detach().item()
        metrics['return'] = torch.amax(returns).detach().item()
        if self.rl_algo_config.use_value:
            value_loss,vf_clipfrac = compute_value_loss(vpreds,returns.to(log_prob.device),old_values.to(log_prob.device),response_mask,self.rl_algo_config.cliprange_value)
            loss = pg_loss + value_loss
            metrics['critic/vf_clipfrac'] = vf_clipfrac.detach().item()
            metrics['train/vf_loss'] = value_loss.detach().item()
            valid_values = torch.masked_select(vpreds, response_mask).cpu()
            valid_returns = torch.masked_select(returns,response_mask.cpu())
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
            metrics['critic/explained_variance']=(1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
        else:
            loss = pg_loss

        if self.rl_algo_config.entropy_bonus is not None:
            entropy = compute_entropy_loss(logits,response_mask)
            entropy_loss = -entropy*self.rl_algo_config.entropy_bonus
            metrics['train/entropy'] = entropy.detach().item()
            loss = loss+entropy_loss

        if ref_log_probs is not None and self.rl_algo_config.kl_coeff is not None:
            kld = compute_full_kl_penalty(log_probs,ref_log_probs.to(log_probs.device))
            metrics['train/ref_kl_divergence'] = kld.mean().item()
            loss = loss + (kld * self.rl_algo_config.kl_coeff).mean()

        if rollout_log_probs is not None:
            kld = compute_full_kl_penalty(log_probs.cpu(),rollout_log_probs.cpu())
            metrics['train/rollout_kl_divergence'] = kld.mean().item()        
        return loss,metrics
    
    def bc_loss(self, log_probs, expert_actions, dagger_mask, label_smoothing=0.1, **kwargs):
        """
        Behavior Cloning / DAgger Loss.
        """
        import torch.nn.functional as F
        
        # Flatten for CrossEntropyLoss
        # log_probs: [B, S, Vocab] -> [B*S, Vocab]
        # expert_actions: [B, S] -> [B*S]
        
        # We only want to train on the specific tokens masked for DAgger
        # (e.g. the 5% worst episodes)
        # --- ROBUSTNESS FIX: Sanitize Targets ---
        # Ensure we don't train on -1 or indices >= vocab size
        vocab_size = log_probs.size(-1)
        dagger_mask = dagger_mask.bool()
        # Create a mask of valid targets
        # This filters out -1s (Oracle failures) or garbage indices
        
        # Combine with the requested DAgger mask
        # We only train if: 1. It's selected for DAgger AND 2. The label is valid
        active_indices = dagger_mask.view(-1).to(log_probs.device)

        flat_log_probs = log_probs.view(-1, log_probs.size(-1))
        flat_targets = expert_actions.view(-1).to(log_probs.device)
        valid_target_mask = (flat_targets >= 0) & (flat_targets < vocab_size)
        active_indices = active_indices & valid_target_mask.to(active_indices.device)
        if not active_indices.any():
            zero_loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
            return zero_loss, {}

        logits = kwargs.get('logits')
        if logits is not None:
            action_logits = logits[..., self.vocab_ids]
            flat_logits = action_logits.view(-1, action_logits.size(-1))
            loss = F.cross_entropy(
            flat_logits[active_indices], 
            flat_targets[active_indices],
            label_smoothing=label_smoothing
            )
        else:
            # Fallback if only log_probs available (no smoothing easily available)
            loss = F.nll_loss(
            flat_log_probs[active_indices], 
            flat_targets[active_indices]
            )
        
        metrics = {'loss/dagger_loss': loss.detach().item()}
        
        # Optional: Scale the loss if needed (usually done in config)
        return loss, metrics
    
    def generic_train_step(self,embeds_inputs,loss_fn_names,loss_kwargs_list,loss_weights=None):
        '''
        Generic train step that can be used for both RL, SFT, or any unholy combination thereof
        '''
        self._setup_training()
        # Accumulate gradients (handle micro-batches)
        with self.accelerator.accumulate(self.ddp_model):
            loss = torch.tensor(0.0).to(self.device)
            metrics = {}
            logits,vpreds = self._training_forward(embeds_inputs)
            log_probs = self._calculate_action_logprobs(logits) # B by S by N_action space
            if loss_weights is None:
                loss_weights = [1.0]*len(loss_fn_names)
            for loss_fn_name,weight in zip(loss_fn_names,loss_weights):
                loss_fn = getattr(self, f"{loss_fn_name}_loss")
                loss_part,metric = loss_fn(log_probs=log_probs,vpreds=vpreds,logits=logits,**loss_kwargs_list[loss_fn_name])
                loss = loss + loss_part*weight
                metrics |= metric
                
            self.accelerator.backward(loss)
            # Clip gradients and return the total norm (Global L2)
            # max_grad_norm is usually 0.5 or 1.0 in PPO papers
            grad_norm = self.accelerator.clip_grad_norm_(
                self.ddp_model.parameters(), 
                max_norm=1.0 
            )
            # Log the norm (Detect explosions if this spikes > 10.0)
            metrics['train/grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            
            self.optimizer.step()
            self.scheduler.step()
            metrics['train/lr'] = self.scheduler.get_last_lr()[0]
            self.optimizer.zero_grad()
        return metrics
            
    def train_rl_step(self,embeds_inputs,actions,old_log_prob,advantages,returns=None,old_values=None,rollout_log_probs=None,ref_log_probs=None):
        '''
        Docstring for train_rl_step
        
        :param embeds_inputs: batch of embeds for forward pass 
        :param old_log_prob: B by S 
        :param advantages: B by S advantages
        :param returns: targets for value head, B by S
        :param rollout_log_prob: Optional for rollout correction (not yet implemented)
        :param ref_logprobs: B by S by Action Space
        '''
        from verl.trainer.ppo.core_algos import compute_value_loss,compute_entropy_loss

        self.ddp_model.train()
        if self.is_merged():
            self.unmerge_adapter()
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
        self.reset() #clear internal state, training is (mostly) stateless
        self.accelerator.wait_for_everyone() # ensure all workers have unmerged before training
        # Accumulate gradients (handle micro-batches)
        with self.accelerator.accumulate(self.ddp_model):
            # Forward via DDP wrapper (triggers sync)
            logits,vpreds = self.ddp_model(embeds_inputs = embeds_inputs,compute_values = self.rl_algo_config.use_value,value_grad_scale=self.rl_algo_config.value_grad_scale)
            log_probs = self._calculate_action_logprobs(logits) # B by S by N_action space
            log_prob = torch.gather(log_probs, -1, actions.unsqueeze(-1).to(log_probs.device)).squeeze(-1)
            
            '''
            old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
            log_prob (torch.Tensor):
                Log-probabilities of actions under the current policy, shape (batch_size, response_length).
            advantages (torch.Tensor):
                Advantage estimates for each action, shape (batch_size, response_length).
            response_mask (torch.Tensor):
                Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
            loss_agg_mode (str, optional):
                Aggregation mode for `agg_loss`. Defaults to "token-mean".
            config: `(verl.trainer.config.ActorConfig)`: config for the actor.
            '''
            
            response_mask = torch.ones_like(log_prob).bool() #TODO: rollout correction, rejection sampling to exclude bad tokens
            pg_loss,metrics = self.policy_loss_fn(old_log_prob=old_log_prob.to(log_prob.device),log_prob=log_prob,advantages=advantages.to(log_prob.device),response_mask=response_mask,config = self.rl_algo_config)
            metrics['loss/pg_loss'] = pg_loss.detach().item()
            metrics['return'] = torch.amax(returns).detach().item()
            if self.rl_algo_config.use_value:
                value_loss,vf_clipfrac = compute_value_loss(vpreds,returns.to(log_prob.device),old_values.to(log_prob.device),response_mask,self.rl_algo_config.cliprange_value)
                loss = pg_loss + value_loss
                metrics['critic/vf_clipfrac'] = vf_clipfrac.detach().item()
                metrics['train/vf_loss'] = value_loss.detach().item()
                valid_values = torch.masked_select(vpreds, response_mask).cpu()
                valid_returns = torch.masked_select(returns,response_mask.cpu())
                return_diff_var = torch.var(valid_returns - valid_values)
                return_var = torch.var(valid_returns)
                metrics['critic/explained_variance']=(1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
            else:
                loss = pg_loss

            if self.rl_algo_config.entropy_bonus is not None:
                entropy = compute_entropy_loss(logits,response_mask)
                entropy_loss = -entropy*self.rl_algo_config.entropy_bonus
                metrics['train/entropy'] = entropy.detach().item()
                loss = loss+entropy_loss

            if ref_log_probs is not None and self.rl_algo_config.kl_coeff is not None:
                kld = compute_full_kl_penalty(log_probs,ref_log_probs.to(log_probs.device))
                metrics['train/ref_kl_divergence'] = kld.mean().item()
                loss = loss + (kld * self.rl_algo_config.kl_coeff).mean()

            if rollout_log_probs is not None:
                kld = compute_full_kl_penalty(log_probs.cpu(),rollout_log_probs.cpu())
                metrics['train/rollout_kl_divergence'] = kld.mean().item()
            # Backward (handles mixed precision scaling)
            self.accelerator.backward(loss)
            # Clip gradients and return the total norm (Global L2)
            # max_grad_norm is usually 0.5 or 1.0 in PPO papers
            grad_norm = self.accelerator.clip_grad_norm_(
                self.ddp_model.parameters(), 
                max_norm=1.0 
            )
            # Log the norm (Detect explosions if this spikes > 10.0)
            metrics['train/grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            
            self.optimizer.step()
            self.scheduler.step()
            metrics['train/lr'] = self.scheduler.get_last_lr()[0]
            self.optimizer.zero_grad()
        return metrics    
    
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
            # unwrapped = self.accelerator.unwrap_model(self.ddp_model)
            # unwrapped.vlm.save_pretrained(path)
            self.model.save_pretrained(path)
            print(f"Adapters saved to {path}")

    def save_adapter_unsafe(self, path):
        """
        Saves ONLY the LoRA adapters. 
        Driver script is responsible for making the other VLM workers stay put, hence "unsafe"
        """
        
        self.model.save_pretrained(path)
        print(f"Adapters saved to {path}")

    def save_checkpoint_unsafe(self, path):
        """
        Ray-Optimized Saver. 
        NO BARRIERS. Call this ONLY on Rank 0 (Worker[0]).
        The Driver script MUST ensure all other workers are idle/waiting 
        via ray.get() before triggering this.
        """
        import os
        os.makedirs(path, exist_ok=True)
        # 1. Save Model (Adapters)
        # Standard DDP models are replicated, so Rank 0 has everything.
        # save_pretrained is a local I/O operation.
        self.model.save_pretrained(path)
        # 2. Save Optimizer & Scheduler
        # In Standard DDP, optimizer states are identical across ranks.
        # Saving Rank 0's copy is sufficient to restore training.
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        print(f"✅ Checkpoint saved to: {path}")

    def load_checkpoint(self, path, strict_base_check=True,load_optim=True,load_sched=False):
        """
        Resumes training state fully. 
        Must be called AFTER setup_training().
        """
        import os
        from peft.utils import set_peft_model_state_dict, load_peft_weights
        from utils.factories import get_base_model
        # 1. Base Model Check
        
        if strict_base_check:
            saved_base = get_base_model(path)
            if  saved_base is None:
                print(f"⚠️ WARNING: no base model name found")
            elif self.model_id not in saved_base and saved_base not in self.model_id:
                print(f"⚠️ WARNING: Checkpoint base '{saved_base}' != Current '{self.model_id}'")
        # 2. Load Weights (Adapters + Value Head)
        # This updates self.model in-place, preserving optimizer references
        if os.path.exists(os.path.join(path, "adapter_model.bin")) or os.path.exists(os.path.join(path, "adapter_model.safetensors")):
             adapter_state_dict = load_peft_weights(path)
             set_peft_model_state_dict(self.model, adapter_state_dict)
             print(" -> Adapters and (maybe) Value Head loaded.")
        else:
             print(" -> ⚠️ No adapter weights found in checkpoint.")

        # 3. Load Optimizer
        opt_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(opt_path) and load_optim:
            print("loading optimizer!")
            opt_state = torch.load(opt_path, map_location=self.accelerator.device)
            self.optimizer.load_state_dict(opt_state)
            print(" -> Optimizer loaded.")
        
        # 4. Load Scheduler
        sched_path = os.path.join(path, "scheduler.pt")
        if os.path.exists(sched_path) and load_sched:
            print("loading scheduler!")
            sched_state = torch.load(sched_path, map_location=self.accelerator.device)
            self.scheduler.load_state_dict(sched_state,weights_only=True)
            print(" -> Scheduler loaded.")

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
        action,_,_ = worker.infer_probs(messages,images)
        # action = worker.infer_step(messages,images)
