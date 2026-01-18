import torch 
from transformers import Cache, Qwen3VLForConditionalGeneration, Qwen3VLTextModel,Qwen3VLModel, Qwen3VLVisionModel
from transformers.utils.generic import check_model_inputs
from typing import Any, Callable, Optional, TypedDict, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
import torch.nn as nn
from transformers import AutoConfig
def load_sparse_model(model_path, **kwargs):
    # 1. Load the Config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. Instantiate our Custom Sparse Class
    # This creates the model with random weights but the correct sparse architecture
    # model = Qwen3VLSparseForConditionalGeneration(config)
    
    # 3. Load Pretrained Weights
    # We use from_pretrained on our class, pointing to the original model directory.
    # Because our attribute names (self.model, self.text_model) match the original,
    # the weights map 1:1.
    model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
        model_path, 
        config=config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation = "flash_attention_2",
        **kwargs
    )
    
    return model

def extract_contiguous_segments(tensor, mask):
    """
    Args:
        tensor: Shape (batch, seq, hidden)
        mask: Boolean tensor of shape (batch, seq)
    
    Returns:
        List[List[Tensor]]: A nested list where results[b] contains 
                            tensors of contiguous true regions for batch b.
    """
    results = []
    batch_size = tensor.shape[0]

    for b in range(batch_size):
        # Slice the current batch
        curr_tensor = tensor[b]
        curr_mask = mask[b]
        
        # 1. Pad the mask with False (0) on both sides.
        #    This ensures we catch segments starting at index 0 or ending at the last index.
        #    Example: [T, T, F] -> [0, 1, 1, 0, 0]
        padded_mask = torch.cat([
            torch.tensor([0], device=mask.device, dtype=torch.int), 
            curr_mask.int(), 
            torch.tensor([0], device=mask.device, dtype=torch.int)
        ])
        
        # 2. compute diff: 
        #    +1 indicates a transition from False to True (Start)
        #    -1 indicates a transition from True to False (End)
        diff = padded_mask[1:] - padded_mask[:-1]
        
        # 3. Get indices where transitions occur
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        
        # 4. Slice the tensor and collect segments
        batch_segments = []
        for s, e in zip(starts, ends):
            batch_segments.append(curr_tensor[s:e])
            
        results.append(batch_segments)

    return results


def filter_embeds_spatial(
    image_embeds, 
    position_id,
    past_image_embeds=None, 
    min_local_keep=30, 
    max_local_keep=None, 
    max_global_keep=None, 
    threshold=0.8
):
    '''
    Filter embeddings based on both spatial proximity (voxel regions) and feature similarity.
    Filters out embeds that are in the same spatial voxel region with similar features.
    This function extends filter_embeds with spatial awareness.
    
    warning: this function assumes all image_embeds are from the same sequence. if your batch size > 1, 
    you must split the global image embeds list by batch.

    :param image_embeds: list of num_image tensors of N_patch_in_image by N_hidden
    :param position_id: Tensor of shape [1, N, 3] where 1 is batch size, 3 is 3D coordinates (x,y,z), 
                        N is total number of patches (sum of all N_i)
    :param past_image_embeds: tensor of N_patch_in_past by N_hidden. MUST BE NORMALIZED
    :param min_local_keep: minimum number of patches kept per image
    :param max_local_keep: maximum number of patches kept per image
    :param max_global_keep: maximum number of patches kept for the entire input sequence
    :param threshold: Similarity threshold (keep if similarity < threshold)
    '''
    import torch
    import torch.nn.functional as F

    device = image_embeds[0].device
    if past_image_embeds is not None:
        past_image_embeds = past_image_embeds.to(device)
    
    # 1. Normalize and Flatten
    # We keep the list 'image_embeds' for the loop, but need a flat version for the DB
    image_embeds = [F.normalize(emb, p=2, dim=1) for emb in image_embeds]
    image_embeds_flat = torch.cat(image_embeds, dim=0)
    
    total_patches = image_embeds_flat.shape[0]
    
    # Extract and process position_id: [1, N, 3] -> [N, 3]
    # Remove batch dimension and transpose to get (N, 3)
    position_3d = position_id[0] # Shape: [N, 3]
    
    # Verify that position_id matches total number of patches
    if position_3d.shape[0] != total_patches:
        raise ValueError(f"position_id length ({position_3d.shape[0]}) must match total number of patches ({total_patches})")
    
    # Convert 3D coordinates to discrete voxel indices (already discrete, just use as-is)
    voxel_indices = position_3d.long()  # Shape: [N, 3]
    
    # Dictionary to track kept embeds per voxel: {voxel_tuple: list of embed indices}
    voxel_to_kept_indices = {}
    
    keep_mask = torch.zeros(total_patches, dtype=torch.bool, device=device)
    
    cursor = 0

    for embeds in image_embeds:
        n_tokens = embeds.shape[0]
        
        # Get voxel indices for current image's embeds
        curr_voxel_indices = voxel_indices[cursor:cursor + n_tokens]  # Shape: [n_tokens, 3]
        
        # --- A. Compute Similarity vs HISTORY (if exists) ---
        # We do this per-image to avoid allocating a massive (Past x Total_Seq) matrix
        if past_image_embeds is not None and past_image_embeds.shape[0] > 0:
            # (N_past, D) @ (D, N_curr) -> (N_past, N_curr) -> Max over past
            hist_sim = (past_image_embeds @ embeds.T).amax(dim=0)
        else:
            hist_sim = torch.zeros(n_tokens, device=device)

        # --- B. Compute Similarity vs LOCAL CONTEXT (from other voxels) ---
        # Slice the flattened array to get all *kept* tokens seen so far in this sequence
        valid_history_mask = keep_mask[:cursor]
        
        if valid_history_mask.any():
            local_db = image_embeds_flat[:cursor][valid_history_mask]
            # (N_local_kept, D) @ (D, N_curr) -> Max over local
            local_sim = (local_db @ embeds.T).amax(dim=0)
        else:
            # First image, or previous images were entirely dropped
            local_sim = torch.zeros(n_tokens, device=device)
        
        # --- C. Compute Similarity vs SAME VOXEL (spatial constraint) ---
        spatial_sim = torch.zeros(n_tokens, device=device)
        
        for local_idx in range(n_tokens):
            global_idx = cursor + local_idx
            curr_voxel = tuple(curr_voxel_indices[local_idx].cpu().tolist())
            
            # Check if there are any kept embeds in the same voxel
            if curr_voxel in voxel_to_kept_indices:
                same_voxel_indices = voxel_to_kept_indices[curr_voxel]
                if len(same_voxel_indices) > 0:
                    # Extract embeddings in the same voxel
                    same_voxel_embeds = image_embeds_flat[same_voxel_indices]  # Shape: [M, D]
                    curr_embed = embeds[local_idx:local_idx+1]  # Shape: [1, D]
                    
                    # Compute similarity with all embeds in the same voxel
                    spatial_sim[local_idx] = (same_voxel_embeds @ curr_embed.T).amax(dim=0).squeeze()
        
        # --- D. Combine all similarity measures ---
        # Token is redundant if it matches History OR Local Context OR Same Voxel
        final_sim = torch.maximum(torch.maximum(hist_sim, local_sim), spatial_sim)
        
        # --- E. Determine Keepers & Apply Local Limits ---
        
        # 1. Identify natural keepers based on threshold
        # (Low similarity = Unique = Keep)
        should_keep = final_sim < threshold
        num_keep = should_keep.sum()
        
        # 2. Calculate target count based on limits
        target_count = num_keep
        
        if min_local_keep is not None:
            target_count = max(target_count, min_local_keep)
        if max_local_keep is not None:
            target_count = min(target_count, max_local_keep)
            
        # Ensure target doesn't exceed actual token count
        target_count = min(target_count, n_tokens)
        
        # 3. If limits force a change, assume "Lowest Similarity" -> "Most Unique/Important"
        if target_count != num_keep:
            # Sort by similarity ascending (0.1, 0.2, ... 0.9)
            # We want to keep the lowest values.
            _, sorted_indices = torch.sort(final_sim, descending=False)
            
            # Reset mask and pick top k unique
            should_keep = torch.zeros_like(should_keep)
            should_keep[sorted_indices[:target_count]] = True

        # --- F. Update Global Mask and Voxel Tracking ---
        keep_mask[cursor : cursor + n_tokens] = should_keep
        
        # Update voxel tracking for kept embeds
        for local_idx in range(n_tokens):
            if should_keep[local_idx]:
                global_idx = cursor + local_idx
                curr_voxel = tuple(curr_voxel_indices[local_idx].cpu().tolist())
                if curr_voxel not in voxel_to_kept_indices:
                    voxel_to_kept_indices[curr_voxel] = []
                voxel_to_kept_indices[curr_voxel].append(global_idx)
        
        cursor += n_tokens

    # --- G. Handle Global Limit ---
    embeds_to_keep_idx = torch.nonzero(keep_mask).squeeze()
    
    if max_global_keep is not None and embeds_to_keep_idx.numel() > max_global_keep:
        # For global limit, we randomly sample to preserve distribution across images
        # (Or you could implement a global priority queue, but random is standard here)
        subset = torch.randperm(embeds_to_keep_idx.numel(), device=device)[:max_global_keep]
        embeds_to_keep_idx = embeds_to_keep_idx[subset]
        embeds_to_keep_idx, _ = embeds_to_keep_idx.sort()

    return embeds_to_keep_idx, image_embeds_flat[embeds_to_keep_idx].detach()

def filter_embeds(
    image_embeds, 
    past_image_embeds=None, 
    min_local_keep=30, 
    max_local_keep=None, 
    max_global_keep=None, 
    threshold=0.8
):
    '''
    warning: this function assumes all image_embeds are from the same sequence. if your batch size > 1, 
    you must split the global image embeds list by batch.

    :param image_embeds: list of num_image tensors of N_patch_in_image by N_hidden
    :param past_image_embeds: tensor of N_patch_in_past by N_hidden. MUST BE NORMALIZED
    :param min_local_keep: minimum number of patches kept per image
    :param max_local_keep: maximum number of patches kept per image
    :param max_global_keep: maximum number of patches kept for the entire input sequence
    :param threshold: Description
    '''
    import torch
    import torch.nn.functional as F

    device = image_embeds[0].device
    if past_image_embeds is not None:
        past_image_embeds = past_image_embeds.to(device)
    # 1. Normalize and Flatten
    # We keep the list 'image_embeds' for the loop, but need a flat version for the DB
    image_embeds = [F.normalize(emb, p=2, dim=1) for emb in image_embeds]
    image_embeds_flat = torch.cat(image_embeds, dim=0)
    
    total_patches = image_embeds_flat.shape[0]
    keep_mask = torch.zeros(total_patches, dtype=torch.bool, device=device)
    
    cursor = 0

    for embeds in image_embeds:
        n_tokens = embeds.shape[0]
        
        # --- A. Compute Similarity vs HISTORY (if exists) ---
        # We do this per-image to avoid allocating a massive (Past x Total_Seq) matrix
        if past_image_embeds is not None and past_image_embeds.shape[0] > 0:
            # (N_past, D) @ (D, N_curr) -> (N_past, N_curr) -> Max over past
            hist_sim = (past_image_embeds @ embeds.T).amax(dim=0)
        else:
            hist_sim = torch.zeros(n_tokens, device=device)

        # --- B. Compute Similarity vs LOCAL CONTEXT ---
        # Slice the flattened array to get all *kept* tokens seen so far in this sequence
        valid_history_mask = keep_mask[:cursor]
        
        if valid_history_mask.any():
            local_db = image_embeds_flat[:cursor][valid_history_mask]
            # (N_local_kept, D) @ (D, N_curr) -> Max over local
            local_sim = (local_db @ embeds.T).amax(dim=0)
            
            # Token is redundant if it matches History OR Local Context
            final_sim = torch.maximum(hist_sim, local_sim)
        else:
            # First image, or previous images were entirely dropped
            final_sim = hist_sim
        # --- C. Determine Keepers & Apply Local Limits ---
        
        # 1. Identify natural keepers based on threshold
        # (Low similarity = Unique = Keep)
        should_keep = final_sim < threshold
        num_keep = should_keep.sum()
        
        # 2. Calculate target count based on limits
        target_count = num_keep
        
        if min_local_keep is not None:
            target_count = max(target_count, min_local_keep)
        if max_local_keep is not None:
            target_count = min(target_count, max_local_keep)
            
        # Ensure target doesn't exceed actual token count
        target_count = min(target_count, n_tokens)
        
        # 3. If limits force a change, assume "Lowest Similarity" -> "Most Unique/Important"
        if target_count != num_keep:
            # Sort by similarity ascending (0.1, 0.2, ... 0.9)
            # We want to keep the lowest values.
            _, sorted_indices = torch.sort(final_sim, descending=False)
            
            # Reset mask and pick top k unique
            should_keep = torch.zeros_like(should_keep)
            should_keep[sorted_indices[:target_count]] = True

        # --- D. Update Global Mask ---
        keep_mask[cursor : cursor + n_tokens] = should_keep
        cursor += n_tokens

    # --- E. Handle Global Limit ---
    embeds_to_keep_idx = torch.nonzero(keep_mask).squeeze()
    
    if max_global_keep is not None and embeds_to_keep_idx.numel() > max_global_keep:
        # For global limit, we randomly sample to preserve distribution across images
        # (Or you could implement a global priority queue, but random is standard here)
        subset = torch.randperm(embeds_to_keep_idx.numel(), device=device)[:max_global_keep]
        embeds_to_keep_idx = embeds_to_keep_idx[subset]
        embeds_to_keep_idx, _ = embeds_to_keep_idx.sort()

    return embeds_to_keep_idx, image_embeds_flat[embeds_to_keep_idx].detach()

class Qwen3VLSparseTextModel(Qwen3VLTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        past_image_embeds = None,
        save_image_db = False,
        **kwarg,
    ):
        # print(input_ids.shape)
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        self.seq_keep_mask = None
        self.vis_keep_mask = None
        # 1. Determine Sequence Length
        if inputs_embeds is not None:
            B, S, H = inputs_embeds.shape
            device = inputs_embeds.device
        elif input_ids is not None:
            B, S = input_ids.shape
            device = input_ids.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # 2. Compute Sparse Mask
        # Default: Keep everything (True)
        final_keep_mask = torch.ones((S,), device=device, dtype=torch.bool,requires_grad=False)

        if inputs_embeds is not None and visual_pos_masks is not None:
            with torch.no_grad():
                # A. Initialize a batch-level mask
                # Shape: (B, S). Init to True. 
                batch_keep_mask = torch.ones((B, S), device=device, dtype=torch.bool)
                ds_keep_mask = torch.zeros(deepstack_visual_embeds[0].shape[0],dtype=torch.bool)
                # B. Mark ALL visual positions as False initially (we will only add back the keepers)
                batch_keep_mask[visual_pos_masks] = False
                
                # C. Extract and Filter
                image_embeds_list = extract_contiguous_segments(inputs_embeds, visual_pos_masks)
                # image_position_ids = extract_contiguous_segments(position_ids.permute(1,2,0), visual_pos_masks)
                # TODO: Implement 'ds_cursor' offset logic here to support Batch Size > 1. 
                # Currently, 'embeds_to_keep_rel_idx' resets to 0 for every batch item, 
                # which causes index collisions in 'ds_keep_mask' if B > 1.
                self.kept_visual_embeds = []
                for b, image_embeds in enumerate(image_embeds_list):
                    if not image_embeds: # Handle cases with no images
                        # TODO: Append empty placeholder to self.kept_visual_embeds to maintain alignment for B > 1
                        continue
                    # Get indices of embeddings to KEEP (relative to the visual segments)
                    embeds_to_keep_rel_idx,filtered_embeds = filter_embeds(image_embeds,past_image_embeds[b] if past_image_embeds is not None else None,max_global_keep=27000,threshold=0.95) #TODO: eliminate magic numbers, previously 0.95
                    # embeds_to_keep_rel_idx,filtered_embeds = filter_embeds_spatial(image_embeds, image_position_ids[b],past_image_embeds[b] if past_image_embeds is not None else None,max_global_keep=27000,threshold=0.95)
                    if save_image_db:
                        self.kept_visual_embeds.append(filtered_embeds.cpu().clone())
                    # MAP RELATIVE INDICES -> GLOBAL INDICES
                    # Get global indices where this batch has visual tokens
                    global_visual_indices = torch.nonzero(visual_pos_masks[b]).squeeze()
                    
                    # Select the global indices corresponding to the kept relative indices
                    global_indices_to_keep = global_visual_indices[embeds_to_keep_rel_idx]
                    
                    # Set these specific visual tokens back to True
                    batch_keep_mask[b, global_indices_to_keep] = True
                    ds_keep_mask[embeds_to_keep_rel_idx] = True
                # D. Union over batch (Keep token if ANY sequence in the batch needs it)
                # This maintains a rectangular tensor shape required by standard attention
                final_keep_mask = batch_keep_mask.any(dim=0) 

                # E. Sparsify Deepstack args if present
                if deepstack_visual_embeds is not None:
                    deepstack_visual_embeds = [
                        v[ds_keep_mask, :] for v in deepstack_visual_embeds
                    ]
                # print(f"keeping {torch.sum(ds_keep_mask)}/{len(ds_keep_mask)}")
        # 3. Apply Mask to Inputs
        # Helper to slice only if tensor is not None
        def apply_mask(t):
            if t is None: return None
            # Handle tensors that might be (B, S, ...) or (B, 1, S, S)
            if t.shape[1] == S:
                return t[:, final_keep_mask]
            return t # Fallback for shapes that don't match S

        input_ids = apply_mask(input_ids)
        if attention_mask is not None:
            if attention_mask.shape[1] > S:
                # 1. Split history (already sparse) from current chunk (dense)
                past_mask = attention_mask[:, :-S]
                curr_mask = attention_mask[:, -S:]
                
                # 2. Apply sparsity mask ONLY to the current chunk
                curr_mask = curr_mask[:, final_keep_mask]
                
                # 3. Recombine
                attention_mask = torch.cat([past_mask, curr_mask], dim=1)
            else:
                # Fallback: If mask length equals chunk length (e.g. first step), filter normally
                attention_mask = attention_mask[:, final_keep_mask]
        position_ids = position_ids[:,:,final_keep_mask]#.detach()
        input_embeds = apply_mask(inputs_embeds)#.detach()
        # input_embeds.requires_grad_()
        visual_pos_masks = apply_mask(visual_pos_masks)#.detach()

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks, 
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwarg,
        )
        self.seq_keep_mask = final_keep_mask.cpu()
        self.vis_keep_mask = ds_keep_mask.cpu()
        return outputs

# --- 2. The Middle Man (The Composite Model) ---
class Qwen3VLSparseModel(Qwen3VLModel):
    """
    Overrides the main model to inject the Sparse Text Backbone.
    Assumes Qwen3VLModel initializes a self.text_model.
    """
    def __init__(self, config):
        super(Qwen3VLModel,self).__init__(config) #grandparent
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLSparseTextModel(config.text_config)
        self.rope_deltas = None
        # Re-run post_init to ensure weights (if initialized randomly) are correct,
        # though usually we load pretrained immediately after.
        self.post_init()

# --- 3. The Top-Level Conditional Generation Model ---
class Qwen3VLSparseForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    The user-facing class. 
    It replaces the internal model with Qwen3VLSparseModel.
    """
    def __init__(self, config):
        # We do NOT call super().__init__(config) because that would instantiate
        # the heavy standard Qwen3VLModel, which we would immediately throw away.
        # Instead, we init the Grandparent (PreTrainedModel) and build manually.
        
        # Init Qwen3VLPreTrainedModel
        super(Qwen3VLForConditionalGeneration, self).__init__(config) 
        
        # Instantiate OUR Sparse Model
        self.model = Qwen3VLSparseModel(config)
        
        # Standard LM Head (same as original)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            labels = labels[:,self.model.language_model.seq_keep_mask]
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
    
        return ModelOutput({
            "loss":loss,
            "logits": logits,
            "past_key_values": outputs.past_key_values,
            "rope_deltas": outputs.rope_deltas,
            "seq_keep_mask": self.model.language_model.seq_keep_mask
        })

