def get_image_token_indices(input_ids, processor):
    """
    Parses input_ids to find the indices of image tokens for each image in the batch.
    
    Args:
        input_ids (torch.Tensor): Shape (batch_size, seq_len)
        tokenizer: The specific Qwen3-VL tokenizer (needed for special token IDs)
        
    Returns:
        list of lists of tensors: A nested structure where output[b][i] is a 1D LongTensor 
                                  containing the sequence indices for the i-th image in the b-th batch item.
    """
    # Retrieve special token IDs for Qwen-VL family
    # Note: Ensure these match your specific Qwen3 tokenizer config usually:
    # <|vision_start|> and <|vision_end|>
    vision_start_id = processor.vision_start_token_id
    vision_end_id = processor.vision_end_token_id
    
    batch_image_indices = []
    images_per_batch = []
    for b_idx, seq in enumerate(input_ids):
        image_indices_in_example = []
        
        # Find all locations of start and end tokens
        start_indices = (seq == vision_start_id).nonzero(as_tuple=True)[0]
        end_indices = (seq == vision_end_id).nonzero(as_tuple=True)[0]
        
        # Sanity check: ensure we have matched pairs
        if len(start_indices) != len(end_indices):
            # Fallback or error handling for truncated sequences
            # In a proper data pipeline, this shouldn't happen if max_len is sufficient
            min_len = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_len]
            end_indices = end_indices[:min_len]

        for start, end in zip(start_indices, end_indices):
                # Store only the start (inclusive) and end (exclusive) of the image patch tokens
                # +1 to skip <|vision_start|>, end is already exclusive of <|vision_end|>
            image_indices_in_example.append((start.item() + 1, end.item()))
            
        batch_image_indices.append(image_indices_in_example)
        images_per_batch.append(len(batch_image_indices))
        # batch_image_indices.append(image_indices_in_example)
    return batch_image_indices,images_per_batch

class SpatialFeatureExtractor:
    def __init__(self, model, layer_index=-1):
        """
        Args:
            model: The Qwen3VLForConditionalGeneration model (or wrapped PeftModel).
            layer_index: The index of the layer to hook.
                         -1 for the final normalization layer (model.model.language_model.norm).
                         0-27 for specific text decoder layers (model.model.language_model.layers).
        """
        self.captured_hidden = None
        self.hook_handle = None
        
        # 1. Navigate to the Text Model
        # We need to handle potential PeftModel wrapping:
        # Structure: [PeftModel -> base_model ->] Qwen3VLForConditionalGeneration -> model -> language_model
        
        root = model
        if hasattr(root, "base_model"):
            root = root.base_model
        
        # Access the internal Qwen3VLModel
        if hasattr(root, "model"):
            qwen_model = root.model
        else:
            # Fallback if passed Qwen3VLModel directly
            qwen_model = root

        # Access the Qwen3VLTextModel
        if hasattr(qwen_model, "language_model"):
            text_model = qwen_model.language_model
        else:
            raise AttributeError(
                "Could not find 'language_model' attribute. "
                "Check if the model architecture matches the Qwen3VL structure provided."
            )

        # 2. Select the Target Layer
        if layer_index == -1:
            # Target: (norm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
            self.target_layer = text_model.norm
        else:
            # Target: (layers): ModuleList(...)
            layers = text_model.layers
            num_layers = len(layers)
            
            # Handle negative indexing (e.g. -6)
            if layer_index < 0:
                layer_index += num_layers
            
            if 0 <= layer_index < num_layers:
                self.target_layer = layers[layer_index]
            else:
                raise ValueError(f"Layer index {layer_index} out of bounds (0 to {num_layers-1})")

        self._register_hook()
    def _register_hook(self):
        def hook_fn(module, args, output):
            # Qwen3VLTextDecoderLayer returns a tuple: (hidden_states, past_key_values, ...)
            # Qwen3VLTextRMSNorm returns a tensor: hidden_states
            if isinstance(output, tuple):
                self.captured_hidden = output[0]
            else:
                self.captured_hidden = output
                
        self.hook_handle = self.target_layer.register_forward_hook(hook_fn)

    def get_and_clear(self):
        if self.captured_hidden is None:
            # This can happen if the hook didn't fire (e.g., during validation if specific steps are skipped)
            # or if called twice. Return None or raise error depending on strictness.
            return None 
            
        tmp = self.captured_hidden
        self.captured_hidden = None 
        return tmp

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()


import torch.nn as nn
import torch
class VLMSpatialHead(nn.Module):
    def __init__(self, hidden_dim=2048, spatial_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, spatial_dim)
        )
        self.spatial_dim = spatial_dim
        self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         # Orthogonal init preserves vector angles better at start
    #         nn.init.orthogonal_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Standard Transformer Init
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # FIX: Do not init bias to 0 for regression. 
                # Init to small random noise prevents p=0 singularity.
                nn.init.normal_(module.bias, mean=0.0, std=0.01)
    def forward(self, hidden_states, batch_image_tuples):
        """
        Args:
            hidden_states: (Batch, Seq_Len, Hidden_Dim)
            batch_image_tuples: List[List[(start, end)]]
        """
        device = hidden_states.device
        
        # --- PHASE 1: Data Gathering (CPU/Pointer Manipulation) ---
        # We use a list comprehension which is faster than explicit for-loops in Python
        # Slicing [b, s:e] creates VIEWS, so this is extremely memory efficient.
        
        # Flatten the structure: get all image patches and their lengths
        flat_patch_views = [
            hidden_states[b_idx, start:end] 
            for b_idx, img_list in enumerate(batch_image_tuples) 
            for (start, end) in img_list
        ]
        
        # If no images in batch, return empty
        if not flat_patch_views:
            return torch.zeros(0, self.spatial_dim, device=device)

        # Calculate lengths for reconstruction later
        # We do this on CPU to avoid CUDA synchronization overhead per item
        lengths = [view.shape[0] for view in flat_patch_views]
        num_images = len(lengths)

        # --- PHASE 2: Heavy Compute (GPU) ---
        
        # 1. Cat: Allocates memory once. Converts ragged views into one dense tensor.
        flat_hidden = torch.cat(flat_patch_views, dim=0) 
        
        # 2. MLP: Run the spatial head on ALL patches from ALL images in parallel
        # This maximizes GPU saturation.
        flat_spatial = self.mlp(flat_hidden) 
        
        # --- PHASE 3: Vectorized Pooling ---
        
        # Instead of a loop to create IDs, we use repeat_interleave.
        # This creates the map: [0, 0, 0, 1, 1, 1, 1, 2, 2 ...] corresponding to image IDs
        lengths_tensor = torch.tensor(lengths, device=device)
        segment_ids = torch.repeat_interleave(
            torch.arange(num_images, device=device), 
            lengths_tensor
        )
        
        # Initialize output container
        pooled_outputs = torch.zeros(num_images, self.spatial_dim, device=device, dtype=flat_spatial.dtype)
        
        # Scatter Add (Sum Pooling)
        # Effectively: pooled[id] += flat_spatial[i]
        pooled_outputs.index_add_(0, segment_ids, flat_spatial)
        
        return pooled_outputs
    

class VLMPoseHead(nn.Module):
    def __init__(self, hidden_dim=2048, spatial_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, spatial_dim)
        )
        self.spatial_dim = spatial_dim
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Orthogonal init preserves vector angles better at start
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def forward(self, hidden_states, batch_image_tuples):
        """
        Args:
            hidden_states: (Batch, Seq_Len, Hidden_Dim)
            batch_image_tuples: List[List[(start, end)]]
        """
        device = hidden_states.device
        
        # --- PHASE 1: Data Gathering (CPU/Pointer Manipulation) ---
        # We use a list comprehension which is faster than explicit for-loops in Python
        # Slicing [b, s:e] creates VIEWS, so this is extremely memory efficient.
        
        # Flatten the structure: get all image patches and their lengths
        flat_patch_views = [
            hidden_states[b_idx, start:end] 
            for b_idx, img_list in enumerate(batch_image_tuples) 
            for (start, end) in img_list
        ]
        
        # If no images in batch, return empty
        if not flat_patch_views:
            return torch.zeros(0, self.spatial_dim, device=device)

        # Calculate lengths for reconstruction later
        # We do this on CPU to avoid CUDA synchronization overhead per item
        lengths = [view.shape[0] for view in flat_patch_views]
        num_images = len(lengths)

        # --- PHASE 2: Heavy Compute (GPU) ---
        
        # 1. Cat: Allocates memory once. Converts ragged views into one dense tensor.
        flat_hidden = torch.cat(flat_patch_views, dim=0) 
        
        # 2. MLP: Run the spatial head on ALL patches from ALL images in parallel
        # This maximizes GPU saturation.
        flat_spatial = self.mlp(flat_hidden) 
        
        # --- PHASE 3: Vectorized Pooling ---
        
        # Instead of a loop to create IDs, we use repeat_interleave.
        # This creates the map: [0, 0, 0, 1, 1, 1, 1, 2, 2 ...] corresponding to image IDs
        lengths_tensor = torch.tensor(lengths, device=device)
        segment_ids = torch.repeat_interleave(
            torch.arange(num_images, device=device), 
            lengths_tensor
        )
        
        # Initialize output container
        pooled_outputs = torch.zeros(num_images, self.spatial_dim, device=device, dtype=flat_spatial.dtype)
        
        # Scatter Add (Sum Pooling)
        # Effectively: pooled[id] += flat_spatial[i]
        pooled_outputs.index_add_(0, segment_ids, flat_spatial)
        
        return pooled_outputs
import torch.nn.functional as F
class PoseRegressionHead(nn.Module):
    def __init__(self, hidden_dim=2048, adapter_dim=1024, output_mlp_dim=1024):
        super().__init__()
        
        # --- 1. Unified Feature Adapter ---
        # Projects to [Geometric_Features (dim) | Attention_Logit (1)]
        self.feature_adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.Mish(), 
            nn.Linear(adapter_dim, adapter_dim + 1) # <--- Optimization here
        )
        
        self.adapter_dim = adapter_dim
        
        # --- 2. Global Regressor ---
        self.global_mlp = nn.Sequential(
            nn.Linear(adapter_dim, output_mlp_dim),
            nn.LayerNorm(output_mlp_dim),
            nn.Mish(),
            nn.Linear(output_mlp_dim, output_mlp_dim // 2),
            nn.Mish(),
            nn.Linear(output_mlp_dim // 2, 7)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states, batch_image_tuples):
        device = hidden_states.device
        
        # --- Phase 1: Gather ---
        flat_patch_views = [
            hidden_states[b_idx, s:e] 
            for b_idx, img_list in enumerate(batch_image_tuples) 
            for (s, e) in img_list
        ]
        
        if not flat_patch_views:
            return torch.zeros(0, 3, device=device), torch.zeros(0, 4, device=device)

        lengths = [v.shape[0] for v in flat_patch_views]
        num_images = len(lengths)
        flat_hidden = torch.cat(flat_patch_views, dim=0) 
        
        # --- Phase 2: Adapt & Split ---
        # Single forward pass for both features and weights
        adapted = self.feature_adapter(flat_hidden)
        
        # Split the last channel
        geometric_features = adapted[:, :-1]      # (N_tokens, dim)
        attn_logits = adapted[:, -1:]             # (N_tokens, 1)
        
        # Sigmoid for weights [0, 1]
        weights = torch.sigmoid(attn_logits)
        
        # --- Phase 3: Weighted Pooling ---
        segment_ids = torch.repeat_interleave(
            torch.arange(num_images, device=device), 
            torch.tensor(lengths, device=device)
        )
        
        # Numerator: sum(w * f)
        numerator = torch.zeros(num_images, self.adapter_dim, device=device, dtype=adapted.dtype)
        numerator.index_add_(0, segment_ids, geometric_features * weights)
        
        # Denominator: sum(w)
        denom = torch.zeros(num_images, 1, device=device, dtype=weights.dtype)
        denom.index_add_(0, segment_ids, weights)
        
        # Normalize (Avoid DivByZero)
        pooled_outputs = numerator / (denom + 1e-6)
        
        # --- Phase 4: Regress ---
        raw_pose = self.global_mlp(pooled_outputs)
        
        pred_t = raw_pose[:, :3]
        pred_q = F.normalize(raw_pose[:, 3:], dim=-1)
        
        return pred_t, pred_q