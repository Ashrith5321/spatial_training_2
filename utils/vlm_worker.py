# import os
import numpy as np
import torch
import time
import gc
import copy
from utils.bev_utils import *
# from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half

class VLMWorker:
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct",attn_implementation='sdpa',dtype='float16', prefix = '<|im_start|>assistant\n**',postfix = '**<|im_end|>',vocab=["stop","forward","left","right","up","down"],cache_outputs=False,load_model=True,offload_cache=False,use_sparse=False,bev_canvas_size=2000,debug_with_humanaction=False):
        import transformers.modeling_flash_attention_utils as fa_utils
        def patched(position_ids, batch_size):
            return False
        fa_utils._is_packed_sequence = patched
        import torch
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.vocab = vocab
        self.vocab_ids = self._vocab_to_ids(vocab)
        self.cache_outputs = cache_outputs
        self.model_id = model_id
        self.attn_implementation = attn_implementation
        self.dtype = dtype
        self.model=None
        self.prefix_ids = self.processor.tokenizer.encode(prefix)
        self.postfix_ids = self.processor.tokenizer.encode(postfix)
        self.offload_cache = offload_cache
        self.use_sparse = use_sparse
        self.bev_canvas_size = bev_canvas_size
        self.debug_with_humanaction = debug_with_humanaction
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
        self.output_list = []
        self.cumulative_inputs = None
        self.seq_keep_mask = None
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
            position_ids, deltas = self.model.model.get_rope_index(
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
            # patch_coords = patch_coords-self.global_min_vals
            w,t,h = patch_coords[...,0],patch_coords[...,1],patch_coords[...,2] # horrific mess here
            w = w.reshape(-1)
            t = t.reshape(-1)
            h = h.reshape(-1)

            patch_coords = torch.stack([t,t+h,t+w],dim=0).reshape(1,3,-1)
            # patch_coords = self.bev_canvas_size//2*torch.ones(1,3,1)
            position_ids = get_pos_id(self.cumulative_inputs['input_ids'],patch_coords.to(self.cumulative_inputs['input_ids'].dtype),self.processor,self.bev_canvas_size)
        return position_ids

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
            if self.cache_outputs:
                self.output_list.append(outputs)
                # Compute logprobs directly (1-to-1 mapping)
            relevant_logits = outputs.logits[0].float()
            scaled_logits = relevant_logits/temperature
            mask = torch.full_like(scaled_logits, float('-inf'))
            mask[:, self.vocab_ids] = 0
            masked_logits = scaled_logits + mask
            all_probs = torch.softmax(masked_logits, dim=-1)
            # all_logprobs = torch.log_softmax(relevant_logits/temperature, dim=-1)
            if self.use_sparse:
                current_keep_mask = self.model.model.language_model.seq_keep_mask
                if self.seq_keep_mask is None:
                    self.seq_keep_mask = current_keep_mask.cpu()
                else:
                    self.seq_keep_mask = torch.cat((self.seq_keep_mask,current_keep_mask.cpu()))
                if self.past_image_embeds is None:
                    self.past_image_embeds = self.model.model.language_model.kept_visual_embeds
                else:
                    for idx, image_embeds in enumerate(self.model.model.language_model.kept_visual_embeds):
                        self.past_image_embeds[idx] = torch.cat((self.past_image_embeds[idx],image_embeds)) #handle the batching...
        if check_probs:
            try:
                assert(torch.argmax(all_probs,dim=-1).item() in self.vocab_ids)
            except:
                print("WARNING: prediction not in provided vocab")
        # print("inference done!")
        if full_logprobs:
            return all_probs.cpu().float().numpy()
        else:
            return all_probs[:,self.vocab_ids].cpu().float().numpy()

    def infer_sequence(self,messages,images,example,full_logprobs=False,temperature=1.2,pos_id_kwargs=None):
        if self.model is None:
            self.load_model()
            self.reset()
        with torch.no_grad():
            seq_inputs = self.tokenize_inputs(messages, images).to(self.device)
            seq_inputs['input_ids'] = seq_inputs['input_ids'][:,:-2]
            seq_inputs['attention_mask'] = seq_inputs['attention_mask'][:,:-2]

            seq_position_ids = self._calculate_seq_pos_id(seq_inputs, example, pos_id_kwargs)
            if pos_id_kwargs is not None and pos_id_kwargs['mode'] == "bev":
                seq_inputs['position_ids'] = seq_position_ids.to(self.device)
            
            seq_image_embeds = self.model.get_image_features(seq_inputs['pixel_values'], seq_inputs['image_grid_thw'])
            seq_outputs = self.model.forward(**seq_inputs, use_cache=True, save_image_db=True)
            # seq_filtered_embeds = self.model.model.language_model.kept_visual_embeds
            seq_cache = seq_outputs['past_key_values']
            relevant_logits = seq_outputs.logits[0].float()
            all_logprobs = torch.log_softmax(relevant_logits/temperature, dim=-1)
            self.seq_outputs = seq_outputs
            self.seq_inputs = seq_inputs
            self.seq_position_ids = seq_position_ids
        if full_logprobs:
            return all_logprobs.cpu().float().numpy()
        else:
            return all_logprobs[:,self.vocab_ids].cpu().float().numpy()
    
    def compare_with_seq(self):
        input_id_diff = (self.cumulative_inputs['input_ids'] - self.seq_inputs['input_ids'].cpu()).abs().sum()
        position_ids_diff = (self.cumulative_inputs['position_ids'] - self.seq_position_ids.cpu()).abs().sum()
        attention_mask_diff = (self.cumulative_inputs['attention_mask'] - self.seq_inputs['attention_mask'].cpu()).abs().sum()
        print('input_id: ', self.cumulative_inputs['input_ids'].shape, self.seq_inputs['input_ids'].shape, input_id_diff)
        print('position_ids: ', self.cumulative_inputs['position_ids'].shape, self.seq_position_ids.shape, position_ids_diff)
        print('attention_mask: ', self.cumulative_inputs['attention_mask'].shape, self.seq_inputs['attention_mask'].shape, attention_mask_diff)
        return

    def infer_probs(self,messages,images,**kwargs):
        # logprobs = self.infer_step(messages,images,**kwargs)
        # assert(len(logprobs)==1) #ensure there is a unique token position for decision making
        # logprobs = logprobs[0]
        # probs = np.exp(logprobs)
        # probs /= np.sum(probs)
        probs = self.infer_step(messages,images,**kwargs)[0]
        return probs
    
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

    class DataGenerator:
        """Generates synthetic turn data."""
        def __init__(self, width, height, processor):
            self.width = width
            self.height = height
            self.processor = processor

        def create_synthetic_image(self):
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

    worker = VLMWorker(model_id=MODEL_ID,attn_implementation='flash_attention_2', dtype='bfloat16',offload_cache=False,use_sparse=True)
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
