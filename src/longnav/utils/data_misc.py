# from datasets import Dataset
import numpy as np
import math
from PIL import Image as PILImage
import io
def get_max_scaling_factor(T: int,N: int=190, M: int=29, H: int=480, W: int=640, K: int=37060) -> float:
    """
    Calculates the maximum image scaling factor to fit a VLM history within a token budget.

    Args:
        N (int): Fixed system prompt cost.
        M (int): Text tokens per turn.
        H (int): Original image height.
        W (int): Original image width.
        T (int): Number of turns (sequence length).
        K (int): Total token budget.

    Returns:
        float: The scaling factor (0.0 to 1.0). Returns 0.0 if impossible.
    """
    # 1. Calculate the budget available for ALL images
    # Total Cost = System + (Text_Per_Turn * Turns) + (Image_Cost * Turns)
    fixed_cost = N + (T * M)
    image_budget_total = K - fixed_cost

    # If we don't even have space for text, return 0
    if image_budget_total < 0:
        return 0.0

    # Calculate max tokens allowed per single image
    max_tokens_per_image = image_budget_total // T

    # If we can't afford even 1 token (1 patch) per image, it's impossible
    if max_tokens_per_image < 1:
        return 0.0

    # 2. Define Cost Function
    # Formula: ceil(h_new / 32) * ceil(w_new / 32)
    def calculate_tokens(scale):
        h_scaled = scale * H
        w_scaled = scale * W
        return math.ceil(h_scaled / 32) * math.ceil(w_scaled / 32)

    # 3. Optimization (Binary Search)
    # We search for the largest S in [0, 1] that satisfies the budget.
    low = 0.0
    high = 1.0
    best_scale = 0.0
    
    # Quick check: does full resolution fit?
    if calculate_tokens(1.0) <= max_tokens_per_image:
        return 1.0

    # Binary search for precision up to 4 decimal places
    for _ in range(20):
        mid = (low + high) / 2
        cost = calculate_tokens(mid)
        
        if cost <= max_tokens_per_image:
            best_scale = mid
            low = mid  # Try to go higher
        else:
            high = mid # Must go lower
    return round(best_scale, 4)


def make_dynamic_resize_transform(N: int=190, M: int=29, H: int=480, W: int=640, K: int=37060,min_visual_tokens=70):
    """
    Calculates the maximum image scaling factor to fit a VLM history within a token budget.

    Args:
        N (int): Fixed system prompt cost.
        M (int): Text tokens per turn.
        H (int): Original image height.
        W (int): Original image width.
        T (int): Number of turns (sequence length).
        K (int): Total token budget.

    Returns:
        float: The scaling factor (0.0 to 1.0). Returns 0.0 if impossible.
    """
    # 3. Define the Transform
    def dynamic_resize_transform(batch):
        """
        Loads images from paths and resizes them based on the episode length.
        """
        # Prepare output list
        batch_images = []
        
        # Iterate over the batch (list of episodes)
        # batch['rgb_sequence'] is a list of lists: [ ["path1", "path2"], ["path1", "path2"] ... ]
        for paths in batch['images']:
            # A. Determine T for this specific episode
            t = len(paths)
            
            # B. Get the optimal scale
            scale = get_max_scaling_factor(
            N=N, 
            M=M, 
            H=H, 
            W=W, 
            T=t, 
            K=K
            )

            min_scale = np.sqrt(min_visual_tokens/(H*W/32/32)) #hard coded, qwen doesn't allow <70 img tok
            if scale < min_scale:
                # print("hit image downscale limit.")
                scale = min_scale
            
            # C. Calculate new integer dimensions
            new_w = int(W * scale)
            new_h = int(H * scale)
            
            # Safety check: if scale is 0 (impossible to fit), we might need to truncate
            # For now, let's assume we maintain at least 1 pixel or handle error
            if new_w < 1 or new_h < 1:
                raise ValueError(f"Episode with length {t} cannot fit in budget {K}!")

            # D. Load and Resize
            episode_imgs = []
            for p in paths:
                # Load from disk
                if isinstance(p, str):
                    img = PILImage.open(p)
                else:
                    img = p 
                # img = PILImage.open(p) 
                
                # Resize only if necessary (Lanczos is best for downsampling)
                if scale < 1.0:
                    img = img.resize((new_w, new_h), resample=PILImage.Resampling.LANCZOS)
                episode_imgs.append(img)
            batch_images.append(episode_imgs)
        # Update the batch. 
        # VLM processors usually expect the column to be named "images" or "pixel_values"
        batch["images"] = batch_images
        return batch
    return dynamic_resize_transform

def filter_corrupt_images(example):
    """
    Tries to open every image in the episode. Returns False if any fail.
    """
    try:
        for path in example['images']:
            # verify() checks headers but doesn't decode pixel data (fast)
            with PILImage.open(path) as img:
                img.verify() 
        return True
    except (OSError, PILImage.UnidentifiedImageError):
        # Print path so you know what's broken
        print(f"Found broken image in episode, removing.") 
        return False



def decode_image_sequence(example):
    # 'images' is a list of dictionaries like {"bytes": b"...", "path": ...}
    # We iterate through the sequence and decode each one
    decoded_sequence = []
    for img_dict in example["images"]:
        if img_dict.get("bytes"):
            # Decode from bytes (standard for streaming parquet/arrow)
            image = PILImage.open(io.BytesIO(img_dict["bytes"]))
        else:
            # Fallback to path if bytes are empty
            image = PILImage.open(img_dict["path"])
            # FORCE RGB CONVERSION
        if image.mode != "RGB":
            image = image.convert("RGB")
        decoded_sequence.append(image)
        
    example["images"] = decoded_sequence
    return example
