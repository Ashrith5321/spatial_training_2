import torch
from dataclasses import dataclass
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
import torch
from utils.bev_utils import *


# Apply to dataset
def mask_user_turns(input_ids, pad_token_id=151643):
    """
    input_ids: Tensor of shape (Batch, Seq_Len)
    Returns: labels Tensor of shape (Batch, Seq_Len) with User/System tokens set to -100.
    """
    # 1. Define the Triggers (Qwen2.5-VL specific based on your probe)
    START_SEQ = torch.tensor([151644, 77091, 198], device=input_ids.device) # <|im_start|> assistant \n
    END_TOKEN = 151645 # <|im_end|>
    
    # 2. Initialize labels as fully masked (-100)
    labels = torch.full_like(input_ids, -100)
    
    batch_size, seq_len = input_ids.shape

    # 3. Iterate rows (We use a loop because "finding sequences" in 2D tensors is complex to vectorize efficiently)
    for i in range(batch_size):
        row = input_ids[i]
        
        # Find all starting positions of the [151644, 77091, 198] sequence
        # We use unfold to create sliding windows of size 3, then check equality
        # shape: (seq_len - 2, 3)
        windows = row.unfold(0, 3, 1) 
        matches = (windows == START_SEQ).all(dim=1).nonzero(as_tuple=True)[0]
        
        for start_idx in matches:
            # The match starts at 'start_idx', so the sequence covers [start_idx, start_idx+1, start_idx+2]
            # The actual content we want to train on starts at start_idx + 3
            content_start = start_idx + 3
            
            # Find the NEXT occurrence of <|im_end|> after this start
            # We slice the row from content_start onwards
            future_tokens = row[content_start:]
            end_offsets = (future_tokens == END_TOKEN).nonzero(as_tuple=True)[0]
            
            if len(end_offsets) > 0:
                # Found an end token for this block
                # The relative offset is end_offsets[0], so absolute index is content_start + offset
                content_end = content_start + end_offsets[0]
                
                # UNMASK the range [content_start : content_end + 1]
                # We include content_end because we WANT the model to predict the EOS token
                labels[i, content_start : content_end + 1] = row[content_start : content_end + 1]
    return labels

def apply_action_dropout(labels, attention_mask, dropout_rate=0.5):
    """
    Randomly drops contiguous chunks of valid labels (Assistant Actions).
    
    Args:
        labels: (Batch, Seq) Tensor. User turns must already be -100.
        attention_mask: (Batch, Seq) Tensor. Modified in-place.
        dropout_rate: Probability of dropping an action chunk.
        
    Returns:
        labels: New labels tensor with dropped chunks set to -100.
    """
    # Clone labels to avoid modifying the original input if that matters, 
    # though usually in a collator we modify in place.
    new_labels = labels.clone()
    
    batch_size, seq_len = labels.shape
    
    for i in range(batch_size):
        row_labels = new_labels[i]
        
        # 1. Find all indices where label is VALID (not -100)
        valid_mask = (row_labels != -100)
        
        # If no actions in this sample, skip
        if not valid_mask.any():
            continue
            
        # 2. Identify Chunks (contiguous regions of valid tokens)
        # We look for transitions from -100 to Valid and Valid to -100
        # diff() works well for this.
        
        # Convert boolean mask to int for edge detection
        valid_int = valid_mask.int()
        
        # Find rising edges (0 -> 1) and falling edges (1 -> 0)
        # Pad with 0 at start/end to handle edge cases
        padded = torch.cat([torch.tensor([0], device=labels.device), valid_int, torch.tensor([0], device=labels.device)])
        diffs = padded.diff()
        
        starts = (diffs == 1).nonzero(as_tuple=True)[0]
        ends = (diffs == -1).nonzero(as_tuple=True)[0]
        
        # Sanity check: starts and ends must correspond 1:1
        assert len(starts) == len(ends)
        
        # 3. Iterate over chunks and Drop
        for start, end in zip(starts, ends):
            # Range is [start, end) because 'end' is the index AFTER the valid chunk
            #skip if we have "**STOP**<im_end>"
            # if torch.equal(new_labels[i,start:end],torch.tensor([   334,  50669,    334, 151645])):
            #     # print("found stop, skipping")
            #     continue
            if torch.rand(1) < dropout_rate:
                # DROP IT
                # 1. Mask Label (Don't train)
                new_labels[i, start:end] = -100
                
                # 2. Mask Attention (Hide from future tokens)
                attention_mask[i, start:end] = 0
                
    return new_labels

@dataclass
class ActionMaskingVLMCollator(DataCollatorForVisionLanguageModeling):
    """
    Extension of TRL's VLM collator that masks ALL User/System tokens in a multi-turn conversation.
    It identifies Assistant turns based on specific start/end token IDs and masks everything else.
    """
    dropout:float=-1
    length_warning: int=40000
    # You must provide these for your specific model (e.g., Llama-3, Qwen)
    def torch_call(self, examples):
        # 1. Let the base class handle the heavy lifting (Image processing, Padding, etc.)
        # This returns a batch with 'input_ids', 'labels', 'pixel_values', 'attention_mask'
        try:
            print(f"DEBUG PRE-COLLATE: {[(len(x['images']), x['images'][0].size) for x in examples if 'images' in x]}")
            
            for x in examples:
                size = x['images'][0].size
                for image in x['images']:
                    assert image.size==size
            # print(f"DEBUG PRE-COLLATE: {[(len(x['messages'])) for x in examples]}")
            # print(examples[0]['images'])
        except Exception as e:
            print(f"collator error! {e}")
            print(examples[0]['images'])
        # print((len(examples[0]['messages'])-1)/2)
        batch = super().torch_call(examples)
        if "image_grid_thw" not in batch:
            print("!!! CRITICAL ERROR: 'image_grid_thw' MISSING in batch! Qwen is blind! !!!")
        else:
            # print(f"debug post collate: thw{batch['image_grid_thw'].shape}")
            pass

        # print(batch['pixel_values'].shape)
        # print(batch['input_ids'].shape)
        # 2. Extract padded input_ids
        # Shape: (Batch_Size, Seq_Len)
        input_ids = batch["input_ids"]

        # print(input_ids.shape)
        # print(batch['pixel_values'].shape)
        if input_ids.shape[1]>=self.length_warning:
            print("warning! sequence too long!")
            print(f"input ids: {batch['input_ids'].shape}")
            # print(f"image shape: {batch['pixel_values'].shape}")
        else:
            print(f"input id shape: {input_ids.shape}")
            # print(f"image shape: {batch['pixel_values'].shape}")

        # 3. Apply our "State Machine" Masking
        # We pass the padded input_ids directly.
        # The function returns a tensor of the same shape with user tokens set to -100.
        labels = mask_user_turns(input_ids)

        if self.dropout>0:
            labels = apply_action_dropout(
                labels, 
                batch["attention_mask"], 
                dropout_rate=self.dropout
            )
        # 4. Handle Padding (Crucial Safety Step)
        # While our function initializes to -100, we should double-check that 
        # actual padding positions (where attention_mask is 0) remain -100.
        if "attention_mask" in batch:
            # Mask out padding explicitly just in case our logic was too aggressive
            labels[batch["attention_mask"] == 0] = -100
            
        # 5. Assign back to batch
        batch["labels"] = labels
        # return batch
        return batch
    
@dataclass 
class PoseVLMCollator(ActionMaskingVLMCollator):

    def torch_call(self, examples):
        
        # 1. Run standard VLM collation (Text + Images + Masking)
        batch = super().torch_call(examples)

        # 2. Extract Geometric Ground Truth
        # We need to extract the pose sequences and flatten them to match 
        # the flattened 'pixel_values' (N_total_images, C, H, W) structure.
        
        all_poses = []
        batch_counts = []
        
        for i, ex in enumerate(examples):
            # Check for standard key names
            pose_seq = ex.get('poses', ex.get('pos_rots'))
            
            if pose_seq is None:
                raise ValueError(f"Example {i} is missing 'poses' (or 'pos_rot') field. "
                                 "Dataset must provide [x,y,z,qx,qy,qz,qw] per image.")
            
            # Convert to tensor: (Num_Images, 7)
            # We treat poses as float32
            poses = torch.tensor(pose_seq, dtype=torch.float32)
            
            # Safety Check: Alignment
            # The number of poses must match the number of images passed to the VLM
            num_imgs = len(ex['images']) if 'images' in ex else 0
            if poses.shape[0] != num_imgs:
                raise ValueError(f"Data Alignment Error in Ex {i}: "
                                 f"Found {num_imgs} images but {poses.shape[0]} poses.")

            all_poses.append(poses)
            batch_counts.append(num_imgs)

        # 3. Stack and Store
        if all_poses:
            # Flatten to (N_total_images_in_batch, 7)
            flat_poses = torch.cat(all_poses, dim=0)
            # print(all_poses)
            # Store Translation
            batch['gt_t'] = flat_poses[:, :3]
            
            # Store Rotation (Normalize to ensure valid unit quaternions)
            # eps added to avoid div-by-zero if data is corrupted (0,0,0,0)
            batch['gt_q'] = torch.nn.functional.normalize(flat_poses[:, 3:], p=2, dim=-1, eps=1e-12)
            
            # Store Batch Counts 
            # (Must be Tensor to survive DataLoader collation, converted to list in Trainer)
            batch['batch_image_counts'] = torch.tensor(batch_counts, dtype=torch.long)
        else:
            print("SEVERE ERROR!!!!")
            # Fallback for text-only batches (rare but possible)
            batch['gt_t'] = torch.empty(0, 3)
            batch['gt_q'] = torch.empty(0, 4)
            batch['batch_image_counts'] = torch.empty(0, dtype=torch.long)

        return batch
#        batch['vp_ids'] = torch.tensor([vp for ex in examples for vp in ex['vp_ids']], dtype=torch.long)


@dataclass 
class PcdVLMCollator(ActionMaskingVLMCollator):
    canvas_size:int=250
    def torch_call(self, examples):

        # 1. Run standard VLM collation (Text + Images + Masking)
        # print(examples)
        keys = ['messages','images']
        batch = super().torch_call([{k:example[k] for k in keys} for example in examples])
        maps = []
        for example in examples:
            # batch['']
            mats = pos_rots_to_matrix(np.array(example['pos_rots'])) @ get_cv_to_habitat_correction()
            # rgbs = np.array(ds[ep_idx]['rgb_sequence'])
            depths = load_depths(example)
          
            divisor = 1
            ps = 32 //divisor
            points = depth_to_pointcloud(depths,fov_degrees=79)
            patch_coords = patch_average_einops(points,patch_size=ps)
            patch_coords_world = transform_points_batch(patch_coords,mats)
            patch_coords_discrete = (patch_coords_world/0.15).astype(int)
            pc_pt = torch.tensor(patch_coords_discrete)
            maps.append(get_thw(pc_pt))

        pos_ids = get_pos_id(batch['input_ids'],torch.stack(maps),self.processor,canvas_size=self.canvas_size)
        batch['position_ids'] = pos_ids

        return batch
#        batch['vp_ids'] = torch.tensor([vp for ex in examples for vp in ex['vp_ids']], dtype=torch.long)
