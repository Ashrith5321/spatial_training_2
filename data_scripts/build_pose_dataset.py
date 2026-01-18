import argparse
import os
import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset, DatasetDict, Features, Value, Sequence
from PIL import Image, UnidentifiedImageError
import textwrap

# ------------------------------------------------------------------------------
# 1. USER PROVIDED LOGIC (Reused)
# ------------------------------------------------------------------------------

def to_convo(example):
   
    # 1. We assume 'images' column exists and is a list of file paths (or objects)
    #    Make sure you access the correct column name (rgb_sequence vs images)
    # image_list = example['rgb_sequence'] 
    import textwrap

    # system_prompt = textwrap.dedent(f"""\
    # You are a visual navigation agent tasked with finding "{example['goal_text']}" in an unknown environment.
    # You will receive a sequence of observations showing your movement history up to the current moment.

    # **Action Space:**
    # [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN]

    # **Your Mission:**
    # 1. Analyze the observation history to understand your current location and orientation.
    # 2. Select the next discrete action to navigate efficiently towards the goal.

    # **Critical Constraints:**
    # * **Collision Detection:** If your previous action was MOVE_FORWARD but the visual observation did not change significantly, you have collided. You MUST turn or move away immediately. Do not keep pushing forward.
    # * **Success Condition:** Output **STOP** ONLY when the target is plainly in view, centered, and within 1 meter (close enough to touch).

    # **Output Format:**
    # Respond with the selected action inside double asterisks.
    # """)
    system_prompt= textwrap.dedent(f"""\
        You are a visual navigation agent tasked with finding "{example['goal_text']}" in an unknown environment.
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
    convo = [
        {"role": "user", "content": [
                # Text Item: explicit None for image
                {"type": "text", "text": system_prompt}, 
        ]},
    ]
    action_mapping = {'STOP': 'stop', 'MOVE_FORWARD': 'forward', 
                        'TURN_LEFT': 'left', 'TURN_RIGHT': 'right', 'LOOK_UP': 'up', 'LOOK_DOWN': 'down'}
    for i, action in enumerate(example['action_sequence']):
        action = action_mapping[action]
        convo += [
             {"role": "user", "content": [
                # Text Item: explicit None for image
                {"type": "text", "text": f"Observation {i}:"}, 

                {"type": "image", "text": None},
                
                # Text Item
                {"type": "text", "text": f"Action {i}:"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": f"**{action}**"}
            ]} 
        ]
    example['messages'] = convo
    return example


def validate_episode_images(example):
    """
    Checks if ALL images in the episode's sequence can be opened.
    Returns False if even one image is broken or missing.
    """
    # We access 'images' because we rename 'rgb_paths' -> 'images' earlier in the pipeline
    image_paths = example.get("images", [])
    
    if not image_paths:
        return False 
    
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify() 
        except:
            return False
    return True

# ------------------------------------------------------------------------------
# 2. MAIN PIPELINE
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build SFT Dataset from Habitat-Web Columnar JSONLs")
    
    parser.add_argument("--input_dir", type=Path, required=True, 
                        help="Directory containing the .jsonl (or .jsonl.parts) files")
    parser.add_argument("--image_root", type=str, required=True, 
                        help="Root directory where images are stored (to make paths absolute)")
    parser.add_argument("--output_dir", type=Path, required=True, 
                        help="Where to save the final HuggingFace dataset")
    
    parser.add_argument("--val_scene_count", type=int, default=5, 
                        help="Number of scenes to hold out for validation (picks scenes with fewest samples)")
    parser.add_argument("--max_length", type=int, default=300, 
                        help="Filter out episodes longer than this")
    parser.add_argument("--num_proc", type=int, default=16, 
                        help="Number of CPU workers for mapping/filtering")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"--- Starting Dataset Build ---")
    print(f"Input: {args.input_dir}")
    print(f"Images: {args.image_root}")
    
    # 1. Load Dataset
    # We glob all .jsonl files in the input directory
    data_files = sorted([str(p) for p in args.input_dir.glob("*.jsonl")])
    if not data_files:
        raise ValueError(f"No .jsonl files found in {args.input_dir}")
    
    print(f"Found {len(data_files)} chunk files. Loading...")
    ds = load_dataset("json", data_files=data_files, split="train")
    
    # 2. Rename Columns (Compatibility)
    # The new JSONL has 'rgb_paths' and 'actions'. 
    # The old logic expects 'images' and 'action_sequence'.
    if "rgb_paths" in ds.column_names:
        ds = ds.rename_column("rgb_paths", "images")
    if "actions" in ds.column_names:
        ds = ds.rename_column("actions", "action_sequence")
        
    print(f"Initial Count: {len(ds)}")

    # 3. Fix Paths (Relative -> Absolute)
    # We do this before filtering so validation checks valid absolute paths
    print("Fixing image paths...")
    def make_absolute(example):
        example["images"] = [os.path.join(args.image_root, p) for p in example["images"]]
        return example
    
    ds = ds.map(make_absolute, num_proc=args.num_proc, desc="Fixing Paths")
    # 3.5. ALIGNMENT CORRECTION (The missing piece)
    print("Applying Temporal Shift (Image[t] -> Action[t+1])...")
    def temporal_shift(example):
        # We need at least 2 steps to form a valid pair
        if len(example["images"]) < 2:
            # Return empty so filter removes it later
            example["images"] = []
            example["pos_rots"] = []
            example["action_sequence"] = []
            return example

        # User Logic: 
        # Action[i] caused Image[i]. 
        # To predict Action[i+1] (Next move), we use Image[i].
        
        # Images/Poses: Drop the last one (Result of last action)
        example["images"] = example["images"][:-1]
        example["pos_rots"] = example["pos_rots"][:-1]
        
        # Actions: Drop the first one (Cause of first image)
        example["action_sequence"] = example["action_sequence"][1:]
        
        return example

    ds = ds.map(temporal_shift, num_proc=args.num_proc, desc="Temporal Shift")
    
    # Filter out empty episodes created by shift
    # ds = ds.filter(lambda x: len(x["images"]) > 0, num_proc=args.num_proc)
    # 4. Filter by Length
    print(f"Filtering episodes > {args.max_length} steps...")
    ds = ds.filter(lambda x: len(x["action_sequence"]) <= args.max_length, 
                   num_proc=args.num_proc, desc="Len Filter")

    # 5. Filter Corrupt Images
    print(f"Validating image integrity (this may take a while)... starting with {len(ds)} samples")
    ds = ds.cast_column('images',Sequence(Value(dtype='string')))
    # ds = ds.filter(validate_episode_images, num_proc=args.num_proc, desc="Img Verify",batch_size=10)
    
    print(f"Valid Count: {len(ds)}")

    # 6. Create Splits (By Scene)
    print("Calculating split based on scene density...")
    # Get all scene IDs
    scene_ids = ds["scene_id"]
    scene_counts = Counter(scene_ids)
    
    # Sort scenes by count (ascending) to find the rarest ones
    # We use rarest scenes for validation to keep Training set data-rich
    sorted_scenes = sorted(scene_counts.items(), key=lambda item: item[1])
    
    # Select N scenes
    val_scenes = set([scene for scene, count in sorted_scenes[:args.val_scene_count]])
    print(f"Selected {len(val_scenes)} validation scenes (Total samples in val: {sum(scene_counts[s] for s in val_scenes)})")
    print(f"Val Scenes: {val_scenes}")

    def is_val(ex):
        return ex["scene_id"] in val_scenes

    def is_train(ex):
        return ex["scene_id"] not in val_scenes

    # Apply split
    # Note: We filter instead of using .train_test_split to ensure strict scene separation
    train_ds = ds.filter(is_train, num_proc=args.num_proc, desc="Split Train")
    val_ds = ds.filter(is_val, num_proc=args.num_proc, desc="Split Val")

    # 7. Apply Formatting (to_convo)
    print("Applying VLM formatting...")
    train_ds = train_ds.map(to_convo, num_proc=args.num_proc, desc="Format Train")
    val_ds = val_ds.map(to_convo, num_proc=args.num_proc, desc="Format Val")

    # 8. Save
    print(f"Saving to {args.output_dir}...")
    final_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds
    })
    
    final_dict.save_to_disk(args.output_dir)
    print("Success.")

if __name__ == "__main__":
    main()

'''
python build_pose_dataset.py \
  --input_dir ~/scratch/qixin/dump/train_all_pose_depth_lt500.jsonl.parts \
  --image_root ~/scratch/qixin/dump/habitat_web_image_depth \
  --output_dir ~/scratch/qixin/dump/sft_datasets/interrim \
  --val_scene_count 8 \
  --max_length 400 \
  --num_proc 16
'''