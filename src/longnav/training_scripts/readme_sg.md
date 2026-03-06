It covers the **Input Specification** (for the data generation team), the **Output Schema** (for the training team), and **Usage Instructions**.

***

# Scene Graph SFT Dataset Builder

This pipeline processes raw navigation episodes into a Visual Language Model (VLM) instruction-tuning dataset. It specifically injects **Scene Graph (SG) Topology** and **Region Localization** information into the training prompts to enhance the agent's spatial awareness.

## 1. Input Data Specification

The pipeline expects input data in **JSON Lines (`.jsonl`)** format. Each line represents a single navigation episode.

### 1.1 JSON Schema
Each JSON object must adhere to the following structure:

```json
{
  "scene_id": "17DRP5sb8fy",          // [String] Unique scene ID (used for train/val splitting)
  "goal_text": "Find the kitchen",    // [String] Navigation instruction
  "images": [                         // [List<String>] Relative paths to RGB images
    "17DRP5sb8fy/000.jpg", 
    "17DRP5sb8fy/001.jpg"
  ],
  "action_sequence": [                // [List<String>] Discrete actions (aligned with images)
    "MOVE_FORWARD", "TURN_LEFT"
  ],
  "scene_graph": {                    // [Dict] The Concise Scene Graph (Details below)
    "rooms": [ ... ]
  }
}
```

### 1.2 The Concise Scene Graph
For training purposes, we require a lightweight SG structure focusing on **Topology** and **Temporal Alignment**.

**Required Structure:**
```json
"scene_graph": {
  "rooms": [
    {
      "id": "0",
      "caption": "apartment",
      "regions": [
        {
          "id": "0.0",              // [Required] Unique ID (Used for Embedding Fusion)
          "caption": "kitchen",     // [Required] Semantic Label (Used for Prompting)
          "image_idx": 5,           // [Required] Frame Index where region is first observed
          "neighbors": ["0.1"]      // [Required] List of connected Region IDs
        },
        {
          "id": "0.1",
          "caption": "living_room",
          "image_idx": 7,
          "neighbors": ["0.0"]
        }
      ]
    }
  ]
}
```

**Constraints:**
1.  **Topology:** `neighbors` IDs must exist within the dataset.
2.  **Alignment:** `image_idx` must be a valid integer index ($0 \le idx < len(images)$).
3.  **No Redundancy:** Please **remove** dense `objects` lists, 3D coordinates, or `clip_features` to optimize I/O performance.

### 1.3 Image Storage
*   **Format:** `.jpg` or `.png`.
*   **Resolution:** Recommended **224x224** or **256x256** (Standard ViT resolution).
*   **Paths:** Store as **relative paths** in the JSONL. The script will prepend the `--image_root` argument to form absolute paths.

---

## 2. Output Data Specification

The script produces a HuggingFace `DatasetDict` saved to disk.

### 2.1 Features
*   `messages`: The formatted VLM conversation (System Prompt + User/Assistant turns).
*   `images`: The list of loaded PIL Images.
*   `region_ids_sequence`: **[New]** A list of Region IDs aligned with the image sequence.
    *   *Example:* `['unknown', '0.0', '0.0', '0.1', ...]`
    *   *Usage:* Used by the training loop to lookup Region Embeddings for feature fusion.

### 2.2 Formatting Logic
The pipeline applies a **Dual-Injection Strategy**:

1.  **System Prompt (Global Topology):**
    > "**Known Environment Topology:** kitchen (0.0) connects to: living_room (0.1)..."
2.  **User Observation (Local State):**
    > "Observation 5 **(Location: kitchen (0.0))**: [Image]..."

---

## 3. Usage

### 3.1 Dependencies
Ensure you have the following libraries installed:
```bash
pip install datasets pillow
```

### 3.2 Running the Builder
Run the script using the following command arguments.

**Arguments:**
*   `--input_dir`: Folder containing `.jsonl` or `.jsonl.parts` files.
*   `--image_root`: Base path to resolve relative image paths in JSONL.
*   `--output_dir`: Path to save the processed HuggingFace dataset.
*   `--val_scene_count`: Number of distinct scenes to hold out for validation (splits by scene ID to prevent leakage).
*   `--max_length`: Filter out episodes longer than $N$ steps.
*   `--num_proc`: Number of CPU workers.

**Example Command:**

```bash
python build_dataset.py \
  --input_dir /Projects/SG_VLN_HumanData/SG-VLN/data/datasets/objectnav/objectnav_mp3d_thda_70k/train_pose.jsonl.parts \
  --image_root /Projects/SG_VLN_HumanData/SG-VLN/data/datasets/objectnav/objectnav_mp3d_thda_70k/objectnav_images \
  --output_dir /Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v1 \
  --val_scene_count 8 \
  --max_length 250 \
  --num_proc 16
```

---

## 4. Methodology & Architecture

The codebase has been refactored following **SOLID principles** to ensure maintainability and ease of experimentation.

### 4.1 Architecture
*   **`DatasetPipeline`**: Handles data loading, path fixing, temporal shifting ($Image_t \rightarrow Action_{t+1}$), and splitting. It is agnostic to the specific prompt format.
*   **`SceneGraphFormatter` (implements `BaseFormatter`)**: Encapsulates the logic for parsing Scene Graphs and constructing prompts.
    *   *Benefit:* If we need to test a "Baseline" (no SG) version later, we simply swap this class without breaking the pipeline.

### 4.2 Alignment Logic (Forward Fill)
Since Scene Graphs are sparse (updated only when entering a new region) but navigation sequences are dense (frame-by-frame), we use a **Bisect (Forward Fill)** algorithm.
*   If the SG updates at Frame 5 (Kitchen) and Frame 10 (Living Room):
*   Frames 5, 6, 7, 8, 9 are all tagged as **Kitchen**.
*   Frame 10 updates to **Living Room**.
*   This ensures the model has continuous state awareness.