from datasets import load_from_disk
import copy
from typing import Optional
import datasets.arrow_dataset
from datasets.arrow_dataset import Dataset
from datasets.fingerprint import fingerprint_transform
# Internal utilities needed for the function logic
from datasets.features.features import _fix_for_backward_compatible_features, FeatureType
from datasets.arrow_dataset import update_metadata_with_features

# 1. Define the patched method
# We keep the defaults at 1000 to match the original signature style, 
# but now we can override them when we call the function.
@fingerprint_transform(inplace=False)
def cast_column_patched(
    self, 
    column: str, 
    feature: FeatureType, 
    new_fingerprint: Optional[str] = None, 
    writer_batch_size=1000, 
    batch_size=1000,
    num_proc = 1
) -> "Dataset":
    """
    Monkeypatched cast_column that accepts batch_size and writer_batch_size.
    """
    feature = _fix_for_backward_compatible_features(feature)
    
    # If it's a decoding feature (like Audio/Image that doesn't change underlying Arrow data immediately)
    if hasattr(feature, "decode_example"):
        dataset = copy.deepcopy(self)
        dataset._info.features[column] = feature
        dataset._fingerprint = new_fingerprint
        dataset._data = dataset._data.cast(dataset.features.arrow_schema)
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        return dataset
    else:
        # THE FIX: This branch now passes the batch sizes down to self.cast()
        features = self.features
        features[column] = feature
        return self.cast(
            features, 
            writer_batch_size=writer_batch_size, 
            batch_size=batch_size,
            num_proc = num_proc
        )

# 2. Apply the patch
print("Applying monkeypatch to Dataset.cast_column...")
datasets.arrow_dataset.Dataset.cast_column = cast_column_patched
print("Patch applied.")

ds = load_from_disk('/Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v2/validation')
from datasets import Image,Sequence
from PIL import Image as PILImage
ds_i = ds.cast_column('images',(Sequence(Image(decode=False))))
def validate_episode_images(example):
    """
    Checks if ALL images in the episode's sequence can be opened.
    Returns False if even one image is broken or missing.
    """
    # We access 'images' because we rename 'rgb_paths' -> 'images' earlier in the pipeline
    image_paths = example.get("images", [])
    # print("hi")
    if not image_paths:
        return False 
    
    for path in image_paths:
        path = path['path']
        try:
            with PILImage.open(path) as img:
                img.verify() 
                # print("verified")
        except:
            # print("failed")
            return False
    return True
def load_image_bytes(batch):
    """
    Explicitly reads file paths into binary bytes.
    Operates on a batch of raw dictionaries: {'bytes': None, 'path': '/...'}
    """
    updated_images_col = []
    
    for row_list in batch['images']:
        new_row = []
        for img_struct in row_list:
            # Check if we have a path and no bytes
            path = img_struct.get('path')
            bytes_data = img_struct.get('bytes')
            
            if path and bytes_data is None:
                try:
                    with open(path, "rb") as f:
                        bytes_data = f.read()
                    # We populate bytes, keeping path for reference (optional)
                    new_row.append({"bytes": bytes_data, "path": path})
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    # Keep original struct on failure
                    new_row.append(img_struct)
            else:
                # Already loaded or empty
                new_row.append(img_struct)
        
        updated_images_col.append(new_row)
    
    return {"images": updated_images_col}


# ds_i = ds_i.filter(validate_episode_images,num_proc=32,batch_size=10)
ds_i = ds_i.cast_column('images',Sequence(Image(decode=False)))
print("2. Manually reading images (Batch Size: 50)...")
# KEY FIX: batch_size=50 ensures we never process >2GB at once.
# writer_batch_size=50 ensures the cache files on disk are also fragmented safely.
ds_embedded = ds_i.map(
    load_image_bytes,
    batched=True,
    batch_size=10, 
    writer_batch_size=32,
    desc="Embedding Images",
    num_proc=32 # Feel free to increase if you have fast disk I/O
)

print("3. Restoring Image feature...")
# Now that 'bytes' are populated, we tell datasets to treat them as images again
final_ds = ds_embedded#.cast_column('images',Sequence(Image()),writer_batch_size=10,batch_size=10,num_proc=8)
# final_ds._writer_batch_size = 10

print("4. Pushing to Hub (skipping internal embedding)...")
# embed_external_files=False is CRITICAL here. 
# It tells the library: "I already did the work, just upload what I gave you."
final_ds.push_to_hub(
    'Aasdfip/habitat_web_pose_val', 
    # max_shard_size='500MB',
    embed_external_files=False 
)


# OUTPUT_DIR = "habitat_shards"
# REPO_ID = "Aasdfip/habitat_web_pose_train"
# ROWS_PER_SHARD = 50  # 50 rows * ~400MB/row = ~20GB raw (safe for writing, compressed is smaller)
#                      # Wait, 50 rows * 200 images * 50KB = ~500MB compressed. 
#                      # This is VERY safe.
# import os,math
# from huggingface_hub import HfApi
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# total_rows = len(final_ds)
# num_shards = math.ceil(total_rows / ROWS_PER_SHARD)

# print(f"Total Rows: {total_rows}")
# print(f"Sharding into {num_shards} files (approx {ROWS_PER_SHARD} rows each)...")

# for shard_idx in range(num_shards):
#     start_idx = shard_idx * ROWS_PER_SHARD
#     end_idx = min(start_idx + ROWS_PER_SHARD, total_rows)
    
#     # Define the slice
#     # select() creates a view, so this is cheap
#     shard = final_ds.select(range(start_idx, end_idx))
    
#     filename = f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
#     filepath = os.path.join(OUTPUT_DIR, filename)
    
#     print(f"Writing {filename} ({start_idx} to {end_idx})...", end="\r")
    
#     # Save this specific slice to Parquet
#     # This handles the Image encoding (PIL -> Bytes) for just these 50 rows
#     shard.to_parquet(filepath)

# print("\n\nAll shards written successfully.")

# print(f"Uploading {OUTPUT_DIR} to {REPO_ID}...")

# api = HfApi()
# api.upload_folder(
#     folder_path=OUTPUT_DIR,
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     path_in_repo="data",  # Standard HF structure puts parquet in 'data/'
#     commit_message="Upload manually sharded Parquet files"
# )

# print("Upload Complete!")