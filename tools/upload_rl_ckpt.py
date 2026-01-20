import argparse
import os
import yaml
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
import subprocess
import tempfile
import sys
import shlex

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_unique_repo_name(api, user_namespace, base_name):
    """
    Checks if repo exists. If so, appends _1, _2, etc. until unique.
    """
    candidate_name = base_name
    counter = 1
    
    while True:
        repo_id = f"{user_namespace}/{candidate_name}"
        try:
            api.repo_info(repo_id)
            # If no exception, repo exists. Update name and retry.
            candidate_name = f"{base_name}_{counter}"
            counter += 1
        except HfHubHTTPError as e:
            # 404 means it doesn't exist, so we can use this name
            if e.response.status_code == 404:
                return candidate_name, repo_id
            raise e

def determine_config_filename(checkpoint_path):
    """
    Determines the filename for the project config to avoid overwriting 
    existing configs in the checkpoint folder.
    """
    base_name = "config.yaml"
    if not (checkpoint_path / base_name).exists():
        return base_name
    
    counter = 1
    while True:
        candidate = f"config_{counter}.yaml"
        if not (checkpoint_path / candidate).exists():
            return candidate
        counter += 1

def get_or_create_collection(api, user_namespace, collection_title):
    """
    Finds a collection by title or creates a new one.
    """
    collections = api.list_collections(owner=user_namespace)
    
    target_collection = None
    for collection in collections:
        if collection.title == collection_title:
            target_collection = collection
            break
            
    if target_collection:
        print(f"✅ Found existing collection: {collection_title}")
        return target_collection.slug
    else:
        print(f"🆕 Creating new collection: {collection_title}")
        new_col = api.create_collection(title=collection_title, namespace=user_namespace)
        return new_col.slug

def sync_config_and_ckpt(user, host, remote_ckpt_str, remote_config_str, local_dest):
    """
    Synchronizes the config and checkpoint directory from a remote host using rsync with --files-from.
    Assumes that the remote paths are single-string-quoted, and relative to the rsync command's base.
    """
    try:
        # Construct the rsync command
        rsync_command = [
            'rsync',
            '-az',
            '--files-from=-',
            f'{user}@{host}:/',  # Ensure to end with a slash for source interpretation
            local_dest,
        ]

        # Prepare the input for --files-from
        paths = f'{remote_ckpt_str}\n{remote_config_str}\n'
        print(f"rsync command: {' '.join(rsync_command)}, stdin: {paths}")  # Debugging: show the command

        # Execute rsync with subprocess
        process = subprocess.Popen(rsync_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=paths)

        # Handle errors
        if process.returncode != 0:
            print(f"rsync failed with return code {process.returncode}")
            print("stdout:", stdout)
            print("stderr:", stderr)
            sys.exit(1)
        else:
            print("rsync completed successfully.")
            print("stdout:", stdout)  # Optional: print rsync output
            print("stderr:", stderr)  # Optional: print rsync errors
    except FileNotFoundError as e:
        print(f"Error: rsync not found. Make sure it's installed and in your PATH: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Hub with Collection organization.")
    parser.add_argument("checkpoint_path", type=Path, help="Path to the specific checkpoint directory")
    parser.add_argument("-r",type=str,default = "yuehu@vino.engin.umich.edu", help = "remote location for file i.e username@hostname")
    parser.add_argument("--name", type=str, default=None, help="Custom name for the repo (overrides config run_name)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    parser.add_argument(
        "--include-optimizer", 
        action="store_true", 
        help="If set, downloads optimizer states. Default is to exclude them to save time/bandwidth."
    )
    args = parser.parse_args()

    # 1. Setup API and User
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info['name']
    except Exception:
        print("❌ Error: You are not logged in. Please run 'huggingface-cli login'.")
        return

    # 2. Resolve Paths
    _temp_dir_handle = None 

    if args.r and args.r!="" and args.r!="local":
        print(f"☁️  Remote mode detected. Fetching from {args.r}...")
        
        # Create a temporary directory
        _temp_dir_handle = tempfile.TemporaryDirectory()
        local_base = Path(_temp_dir_handle.name)
        remote_host = args.r
        
        # Determine remote paths (assuming standard structure on remote)
        # We treat the input path as a string to manipulate it for rsync
        remote_ckpt_str = str(args.checkpoint_path).rstrip('/')
# ---------------------------------------------------------
        # A. & B. Robust Single-Pass Rsync (Filter Method)
        # ---------------------------------------------------------
        # Downloads disjoint files (config + specific checkpoint) in ONE 
        # connection using include/exclude filters.
        
        remote_ckpt_path = Path(remote_ckpt_str)
        # Assuming standard structure: project/checkpoints/ckpt_name -> root is 2 parents up
        remote_project_root = remote_ckpt_path.parent.parent
        
        # Calculate paths relative to the project root
        # e.g., rel_ckpt="checkpoints/checkpoint_191", rel_config="config.yaml"
        rel_ckpt = remote_ckpt_path.relative_to(remote_project_root)
        rel_config = Path("config.yaml")
        
        # Build the rsync command
        # We start with the base command
        rsync_cmd = ["rsync", "-az", "--info=progress2"]
        
        # 1. Include the config file
        rsync_cmd.append(f"--include={rel_config}")
        
        # 2. Include the parent directories of the checkpoint 
        # (Must explicitly include 'checkpoints/' so rsync can traverse down)
        current_part = Path("")
        for part in rel_ckpt.parts[:-1]:
            current_part = current_part / part
            rsync_cmd.append(f"--include={current_part}/")
        if not getattr(args, "include_optimizer", False):
            rsync_cmd.append(f"--exclude={rel_ckpt}/optimizer*")   
        # 3. Include the specific checkpoint and ALL its contents recursively (***)
        rsync_cmd.append(f"--include={rel_ckpt}/***")
        
        # 4. Exclude everything else in the project root
        rsync_cmd.append("--exclude=*")
        
        # 5. Source (project root) and Destination
        # Note: Trailing slash on source ensures we sync relative to root
        rsync_cmd.append(f"{remote_host}:{remote_project_root}/")
        rsync_cmd.append(str(local_base))

        print(f"📥 Downloading checkpoint and config from {remote_host}...")
        try:
            subprocess.run(rsync_cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Error: Rsync failed. Check paths and permissions.")
            sys.exit(1)

        # ---------------------------------------------------------
        # Update Local Paths
        # ---------------------------------------------------------
        # Rsync recreated the relative structure inside local_base.
        # Structure: /tmp/.../checkpoints/checkpoint_191
        ckpt_path = local_base / rel_ckpt
        
        # Structure: /tmp/.../config.yaml
        config_path = local_base / rel_config
    else:
        # Standard Local Logic
        ckpt_path = args.checkpoint_path.resolve()
        if not ckpt_path.exists():
            print(f"❌ Checkpoint path not found: {ckpt_path}")
            return

        # Calculate config relative to local checkpoint
        project_path = ckpt_path.parent.parent
        config_path = project_path / "config.yaml"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint path not found: {ckpt_path}")
        return

    # Assume structure: project_path/checkpoints/checkpoint_name
    # So project_path is 2 levels up
    project_path = ckpt_path.parent.parent
    config_path = project_path / "config.yaml"

    if not config_path.exists():
        print(f"❌ Project config not found at: {config_path}")
        return

    # 3. Read Config
    print(f"📖 Reading config from {config_path}...")
    config_data = load_yaml(config_path)
    
    try:
        task_config = config_data.get('task', {})
        run_name = task_config.get('run_name', ckpt_path.name) # Fallback to folder name
        wandb_project = task_config.get('wandb_project', 'Uncategorized')
    except AttributeError:
        print("❌ Config format invalid. Expected 'task' key.")
        return

    # Override run_name if custom name provided
    ckpt_name = Path(remote_ckpt_str).name
    final_repo_name = args.name if args.name else f"{run_name}-{ckpt_name}"
    # 4. Prepare Repo Name (Overwrite Prevention)
    print(f"🔍 Checking availability for repo name '{final_repo_name}'...")
    unique_name, repo_id = get_unique_repo_name(api, username, final_repo_name)
    
    if unique_name != final_repo_name:
        print(f"⚠️  Repo '{final_repo_name}' exists. Renaming to '{unique_name}'.")

    # 5. Create Repo
    print(f"🚀 Creating repository: {repo_id}")
    create_repo(repo_id, private=args.private, exist_ok=True)

    # 6. Upload Checkpoint Files
    print(f"⬆️  Uploading checkpoint files from {ckpt_path}...")
    api.upload_folder(
        folder_path=ckpt_path,
        repo_id=repo_id,
        repo_type="model"
    )

    # 7. Upload Project Config (Renaming Logic)
    # We check what is LOCALLY in the checkpoint folder to decide the remote name
    target_config_name = determine_config_filename(ckpt_path)
    
    print(f"📄 Uploading project config as '{target_config_name}'...")
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo=target_config_name,
        repo_id=repo_id,
    )

    # 8. Handle Collection
    print(f"🗂  Processing Collection for project '{wandb_project}'...")
    collection_slug = get_or_create_collection(api, username, wandb_project)
    
    print(f"🔗 Adding {repo_id} to collection...")
    try:
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=repo_id,
            item_type="model"
        )
        print("✅ Success!")
    except Exception as e:
        if "already exists" in str(e):
             print("ℹ️  Item already in collection.")
        else:
            print(f"⚠️  Could not add to collection: {e}")

    print(f"\n🎉 Done! View your model at: https://huggingface.co/{repo_id}")
    print(f"📂 View collection at: https://huggingface.co/collections/{collection_slug}")

if __name__ == "__main__":
    main()