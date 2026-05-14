from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import torch
import datasets
from torch.utils.data import DataLoader
from functools import partial
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import AutoConfig, Trainer
from utils.modeling import Qwen3VLSparseForConditionalGeneration
import os
import sys
import argparse
from pathlib import Path
from PIL import Image as PILImage
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
            with PILImage.open(path) as img:
                img.verify() 
        except:
            return False
    return True


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.modules["vllm"] = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTTrainer, SFTConfig
from utils.trainers import PrunedSFTTrainer
from peft import LoraConfig

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# --- CONFIGURATION ---
MODEL_ID = "Aasdfip/qwen3_webnav_0.1" #,"Qwen/Qwen3-VL-2B-Instruct" # Or your specific VLM backbone
TARGET_SEQ_LEN = 1024                  # The 'L' we are testing
BATCH_SIZE = 1                         # The 'B' we are testing
GRADIENT_CHECKPOINTING = True          # Standard for VLM/LLM training
USE_FLASH_ATTN = True                  # Highly recommended for A100


TRAIN_DATASET_DIR = None
EVAL_DATASET_DIR = None
OUTPUT_DIR = "./dump/uniform_400step"
EVAL_MAX_SAMPLES = 40

def get_peak_memory_gb():
    """Helper to get peak GPU memory in GB"""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def preprocess_logits_for_metrics(logits, labels):
    """
    Reduces logits to argmax predictions on GPU to save memory.
    """
    if isinstance(logits, tuple):
        # Depending on model/config, logits might be (logits, loss) or similar
        logits = logits[0]
    
    # Argmax on GPU, return integer IDs
    return logits.argmax(dim=-1)


from utils.training_utils import SpatialFeatureExtractor, get_image_token_indices
from utils.pose import SfMPoseLoss
class PoseTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hook is ephemeral, so we keep it here. 
        # Layer -6 is a good choice (High semantics, but before final reasoning).
        self.spatial_extractor = SpatialFeatureExtractor(self.model, layer_index=-12)
        self.pose_loss = SfMPoseLoss()
        self.loss_weights = {
        "loss_scale": 0.015,  # Down-weight the noisy scale loss
        "loss_grav": 1.0,    # Gravity is a regularization, keep it small
        "loss_rot": 1.0,     # Explicitly stating 1.0 is fine for clarity
        "loss_trans": 0.7
    }

    def create_optimizer(self):
        """
        Setup the optimizer with two parameter groups:
        1. Spatial Head: High Learning Rate (e.g., 1e-3)
        2. Backbone (LoRA): Low Learning Rate (e.g., 3e-5)
        """
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            spatial_head_params = []
            
            # 1. Separate Parameters
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Check if it belongs to our new head
                if "spatial_head" in name:
                    spatial_head_params.append(param)
                else:
                    # Standard Weight Decay logic for Backbone
                    if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                        no_decay_parameters.append(param)
                    else:
                        decay_parameters.append(param)

            optimizer_grouped_parameters = [
                {
                    "params": spatial_head_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": 1e-3,  # <--- CRITICAL: High LR for the scratch head
                },
                {
                    "params": decay_parameters,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate, # The global args.learning_rate (3e-5)
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]
            # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
        return self.optimizer
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        # 1. Pop custom args
        gt_t = inputs.pop('gt_t')
        gt_q = inputs.pop('gt_q')
        if 'batch_image_counts' in inputs:
            batch_image_counts = inputs.pop('batch_image_counts').tolist()
        else:
            batch_image_counts = []
        # 2. Standard LM Loss
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        # 3. Extract Hidden State
        base_model = model.module if hasattr(model, 'module') else model
        intermediate_hidden = self.spatial_extractor.get_and_clear()
        
        # 4. Spatial Forward
        batch_image_indices, _ = get_image_token_indices(inputs['input_ids'], self.processing_class)
        pred_t, pred_q = base_model.spatial_head(intermediate_hidden, batch_image_indices)
        pred_t = pred_t.float()
        pred_q = pred_q.float()
        # 5. Compute Spatial Loss Components
        loss_components_dict = self.pose_loss.forward(pred_t, pred_q, gt_t, gt_q, batch_image_counts)
        
        # Sum components
        spatial_loss_scalar = sum(
            val * self.loss_weights.get(key, 1.0) 
            for key, val in loss_components_dict.items()
        )        
        total_loss = loss + spatial_loss_scalar

        # 6. Detailed Metric Logging (Train AND Eval)
        mode = "train" if self.model.training else "eval"
        
        # Log total spatial loss
        if "spatial loss" not in self._metrics[mode]:
            self._metrics[mode]["spatial loss"] = []
        self._metrics[mode]["spatial loss"].append(spatial_loss_scalar.item())
        
        # Log individual components
        for k, v in loss_components_dict.items():
            if k not in self._metrics[mode]:
                self._metrics[mode][k] = []
            self._metrics[mode][k].append(v.item())

        return (total_loss, outputs) if return_outputs else total_loss

from utils.collators import ActionMaskingVLMCollator,PcdVLMCollator
from utils.data_misc import make_dynamic_resize_transform
SYSTEM_TOKENS = 190
TURN_TOKENS = 33
ORIG_H = 480
ORIG_W = 640
TOTAL_BUDGET = 39000#34000
dynamic_resize_transform = make_dynamic_resize_transform(SYSTEM_TOKENS,TURN_TOKENS,ORIG_H,ORIG_W,TOTAL_BUDGET-600)

def parse_args():
    p = argparse.ArgumentParser(description="SFT training (minimal CLI: only overrides hardcoded paths)")
    p.add_argument("--model_id", type=str, default=MODEL_ID, help="HF model id or local path")
    p.add_argument("--train_dataset_dir", type=str, default=TRAIN_DATASET_DIR, help="load_from_disk() dir")
    p.add_argument(
        "--eval_dataset_dir",
        type=str,
        default=EVAL_DATASET_DIR,
        help="load_from_disk() dir (use empty string to disable eval)",
    )
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Trainer output_dir")
    p.add_argument("--eval_max_samples", type=int, default=EVAL_MAX_SAMPLES, help="Eval subset size")
    p.add_argument("--print_config", action="store_true", help="Print config then exit")
    p.add_argument("--resume_path", type=str, default="",help = "checkpoint to resume")
    return p.parse_args()


def main():
    args = parse_args()

    if args.print_config:
        print(vars(args))
        return

    print(f"--- Starting Memory Profile ---")
    print(f"Model: {args.model_id}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # 1. Load Model in bfloat16 (No quantization)
    print("Loading model...")
    # model = AutoModelForImageTextToText.from_pretrained(
    #     args.model_id,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2", # Changed from flash_attention_2 to sdpa to avoid extra dependencies
    #     # device_map="auto",
    #     # Force single-GPU placement
    #     device_map={"": 0} if torch.cuda.is_available() else None,
    #     # use_cache=False # Important for training with gradient checkpointing
    # )

    
    # 2. Instantiate our Custom Sparse Class
    # This creates the model with random weights but the correct sparse architecture
    # model = Qwen3VLSparseForConditionalGeneration(config)
    
    # 3. Load Pretrained Weights
    # We use from_pretrained on our class, pointing to the original model directory.
    # Because our attribute names (self.model, self.text_model) match the original,
    # the weights map 1:1.
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
        args.model_id, 
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation = "flash_attention_2",
    )
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Setup LoRA
    print("Applying LoRA...")

    peft_config = LoraConfig(
                r=128, lora_alpha=256, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # modules_to_save=["multi_modal_projector"],
                modules_to_save=["spatial_head"], 
    )

    # Load data from HF
    from datasets import load_from_disk, load_dataset, Sequence, Image,Value
    from utils.data_misc import decode_image_sequence
    use_streaming = True
    if not os.path.exists(os.path.expanduser(args.train_dataset_dir)) and "/" in args.train_dataset_dir:
        print(f"Loading Train (Streaming): {args.train_dataset_dir}")
        train_dataset = load_dataset(args.train_dataset_dir, split="train", streaming=use_streaming)
        # IterableDataset needs .shuffle with buffer_size
        if use_streaming:
            train_dataset = train_dataset.shuffle(seed=42, buffer_size=1000)
            train_dataset = train_dataset.map(decode_image_sequence)
            # print(f"length of images in sample before dynamic resize: {len(next(iter(train_dataset))['images'])}")
            # train_dataset = train_dataset.map(dynamic_resize_transform, batched=True,batch_size=1)
            # new_features = train_dataset.features.copy()
            # new_features["images"] = Sequence(Image())

            # 3. Resize
            train_dataset = train_dataset.map(
                dynamic_resize_transform, 
                batched=True, 
                batch_size=1,
                # features=new_features # Explicitly pass the new schema
            )
            
            # print(f"length of images in sample after dynamic resize: {len(next(iter(train_dataset))['images'])}")
        else:
            train_dataset = train_dataset.shuffle(seed=42)
            train_dataset = train_dataset.cast_column("images", Sequence(Image(decode=True)))
            # train_dataset.set_transform(dynamic_resize_transform)

        # train_dataset = train_dataset.filter(lambda x: len(x['images']) <= 70)
        # IterableDataset doesn't support set_transform directly, use map

    else:
        print(f"Loading Train (Disk): {args.train_dataset_dir}")
        train_dataset = load_from_disk(args.train_dataset_dir)
        # train_dataset = train_dataset.cast_column('images',Sequence(Value(dtype='string')))
        # train_dataset = train_dataset.filter(validate_episode_images, num_proc=32, desc="Img Verify",batch_size=10)

        # train_dataset = train_dataset.filter(lambda example:len(example['action_sequence'])>396,batch_size=10,writer_batch_size=10,num_proc=16)
        train_dataset = train_dataset.cast_column("images", Sequence(Image(decode=True)))
        # train_dataset.set_transform(dynamic_resize_transform)
    if args.eval_dataset_dir:
        if not os.path.exists(os.path.expanduser(args.eval_dataset_dir)) and "/" in args.eval_dataset_dir:
            print(f"Loading Eval (Streaming): {args.eval_dataset_dir}")
            # Try to load 'validation' split, or fallback to 'train' if needed (user provided specific val repo)
            try:
                eval_dataset = load_dataset(args.eval_dataset_dir, split="validation", streaming=use_streaming)
            except:
                eval_dataset = load_dataset(args.eval_dataset_dir, split="train", streaming=use_streaming)
            
            if use_streaming:
                eval_dataset = eval_dataset.map(decode_image_sequence)
                print(f"length of images in sample before dynamic resize: {len(next(iter(eval_dataset))['images'])}")
                eval_dataset = eval_dataset.map(dynamic_resize_transform, batched=True,batch_size=1)
                print(f"length of images in sample: {len(next(iter(eval_dataset))['images'])}")
            else:
                pass
                # eval_dataset.set_transform(dynamic_resize_transform)
            # For eval, we need a finite number of samples
            if args.eval_max_samples:
                eval_dataset = eval_dataset.take(args.eval_max_samples)
            
        else:
            print(f"Loading Eval (Disk): {args.eval_dataset_dir}")
            eval_dataset = load_from_disk(args.eval_dataset_dir)
            eval_dataset = eval_dataset.cast_column('images',Sequence(Value(dtype='string')))

            if args.eval_max_samples is not None and args.eval_max_samples > 0:
                eval_dataset = eval_dataset.select(range(min(args.eval_max_samples, len(eval_dataset))))
            eval_dataset = eval_dataset.filter(validate_episode_images, num_proc=32, desc="Img Verify",batch_size=10)

            eval_dataset = eval_dataset.cast_column("images", Sequence(Image()))
            # eval_dataset.set_transform(dynamic_resize_transform)
    else:
        eval_dataset = None
    # eval_dataset = None

    # 4. Training Arguments
    training_args = SFTConfig(
        accelerator_config={
            "dispatch_batches":False
        },
        output_dir=args.output_dir,
        # run_name="qwen-vln-action-dropout",
        save_strategy="steps",        # Save checkpoints frequently
        save_steps=100,
                  # Save every 500 steps
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100,               # Frequency: Every 100 steps
        per_device_eval_batch_size=1,
        
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        logging_steps=1,
        max_length=None,#TARGET_SEQ_LEN,
        packing=False, # FALSE is critical to strictly enforce batch_size x seq_len shape
        bf16=True,     # Use bfloat16 for A100
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=60000,   # We only need a few steps to hit peak memory
        report_to="wandb",
        # dataset_text_field="text",

        resume_from_checkpoint=args.resume_path,
        assistant_only_loss=False,
        optim="paged_adamw_8bit",
        dataloader_num_workers=2,

        remove_unused_columns=False,
        # resume_from_checkpoint='/Projects/SG_VLN_HumanData/contrastive_training_5view_mlp/checkpoint-4050'
        # processing_class = processor
    )

    # 5. Initialize Trainer
    trainer = PrunedSFTTrainer(
        model=model,
        data_collator=PcdVLMCollator(
            processor=processor,
            length_warning = TOTAL_BUDGET,
            # max_length=TOTAL_BUDGET,
            dropout=0.6,
        ),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        # callbacks=[LRSanityCheckCallback(),GradientDebugCallback()],
        # compute_metrics=compute_metrics_wrapper,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # tokenizer=tokenizer,
    )
    if hasattr(trainer.model, "base_model"):
        if hasattr(trainer.model.base_model, "spatial_head"):
            for param in trainer.model.base_model.spatial_head.parameters():
                param.requires_grad = True
    # 6. Run Training & Measure
    torch.cuda.reset_peak_memory_stats()
    print("Starting training loop...")
    resume = (args.resume_path != "")
    if resume:
        print(f"resuming from {args.resume_path}")
    else:
        print("training from scratch")
    trainer.train(resume_from_checkpoint = args.resume_path if resume else False)
    # try:
        
    # except Exception as e:
    #     print(f"\n[!] Training interrupted (likely OOM or interrupt): {e}")
    
    peak_mem = get_peak_memory_gb()
    print(f"\n" + "="*30)
    print(f"RESULTS for B={BATCH_SIZE}, L={TARGET_SEQ_LEN}")
    print(f"Peak VRAM Used: {peak_mem:.2f} GB")
    print(f"="*30)

if __name__ == "__main__":
    main()
