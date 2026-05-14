import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainingArguments,AutoProcessor
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset,load_dataset
from PIL import Image as PILImage
import os
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# --- CONFIGURATION ---
MODEL_ID = "/Projects/SG_VLN_HumanData/SG-VLN/sft_pipeline/text_adapted_model"#,"Qwen/Qwen3-VL-2B-Instruct" # Or your specific VLM backbone
TARGET_SEQ_LEN = 1024                  # The 'L' we are testing
BATCH_SIZE = 1                         # The 'B' we are testing
GRADIENT_CHECKPOINTING = True          # Standard for VLM/LLM training
USE_FLASH_ATTN = True                  # Highly recommended for A100

SYSTEM_TOKENS = 190
TURN_TOKENS = 30
ORIG_H = 480
ORIG_W = 640
TOTAL_BUDGET = 32000#34000

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

from utils.training_utils import SpatialFeatureExtractor,get_image_token_indices
from utils.pose import SfMPoseLoss
class PoseTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hook is ephemeral, so we keep it here. 
        # Layer -6 is a good choice (High semantics, but before final reasoning).
        self.spatial_extractor = SpatialFeatureExtractor(self.model, layer_index=-12)
        self.pose_loss = SfMPoseLoss()
        self.loss_weights = {
        "loss_scale": 0.02,  # Down-weight the noisy scale loss
        "loss_grav": 0.0,    # Gravity is a regularization, keep it small
        "loss_rot": 0.5,     # Explicitly stating 1.0 is fine for clarity
        "loss_trans": 0.8
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
                    "lr": 1e-4,  # <--- CRITICAL: High LR for the scratch head #1e-3 default
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

from utils.collators import PoseVLMCollator
from utils.data_misc import make_dynamic_resize_transform
from utils.training_utils import PoseRegressionHead
SYSTEM_TOKENS = 190
TURN_TOKENS = 30
ORIG_H = 480
ORIG_W = 640
TOTAL_BUDGET = 32000#34000
dynamic_resize_transform = make_dynamic_resize_transform(SYSTEM_TOKENS,TURN_TOKENS,ORIG_H,ORIG_W,TOTAL_BUDGET-600)
def main():
    print(f"--- Starting Memory Profile ---")
    print(f"Model: {MODEL_ID}")
    print(f"Target Seq Len: {TARGET_SEQ_LEN}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # 1. Load Model in bfloat16 (No quantization)
    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if USE_FLASH_ATTN else "eager",
        device_map="auto",
        # use_cache=False # Important for training with gradient checkpointing
    )
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
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
    model.spatial_head = PoseRegressionHead().to(model.device).to(torch.bfloat16)
    # 2. MANUALLY LOAD HEAD WEIGHTS (The Fix)
    # Check if we are resuming from a checkpoint provided in args
    # (You might need to parse args.resume_from_checkpoint or set a variable)
    resume_path = "/Projects/SG_VLN_HumanData/spatial_training/dump/pose_training_test/checkpoint-9500" # Or None if starting fresh

    if os.path.exists(resume_path):
        from safetensors.torch import load_file
        print(f"\n☢️  NUCLEAR LOAD: Forcing head weights from {resume_path}...")
        
        # 1. Take a fingerprint of current (random) weights
        before_fingerprint = model.spatial_head.global_mlp[0].weight.sum().item()
        
        # 2. Load State Dict
        if os.path.exists(os.path.join(resume_path, "adapter_model.safetensors")):
            state_dict = load_file(os.path.join(resume_path, "adapter_model.safetensors"))
        else:
            state_dict = torch.load(os.path.join(resume_path, "adapter_model.bin"), map_location="cpu")
            
        # 3. Clean Keys (The most likely culprit)
        # PEFT saves as "base_model.model.spatial_head..."
        # We need "mlp.0.weight" etc.
        head_sd = {}
        for k, v in state_dict.items():
            if "spatial_head" in k:
                # Strip all possible prefixes to get to the module root
                clean_k = k.replace("base_model.model.spatial_head.", "")
                clean_k = clean_k.replace("spatial_head.", "") 
                head_sd[clean_k] = v
        
        if len(head_sd) == 0:
            raise ValueError("Nuclear Load Failed: No spatial_head keys in checkpoint!")

        # 4. Load
        missing, unexpected = model.spatial_head.load_state_dict(head_sd, strict=True)
        
        # 5. Verify Fingerprint
        after_fingerprint = model.spatial_head.global_mlp[0].weight.sum().item()
        
        if before_fingerprint == after_fingerprint:
            raise RuntimeError("CRITICAL: Weights did not change after loading! Is the checkpoint empty/random?")
            
        print(f"✅ Nuclear Load Success.")
        print(f"   Fingerprint changed: {before_fingerprint:.4f} -> {after_fingerprint:.4f}")
        print(f"   Missing Keys: {missing}")
        print(f"   Unexpected Keys: {unexpected}\n")
    from datasets import load_from_disk
    train_dataset = load_from_disk("/Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v1/train") 
    train_dataset.set_transform(dynamic_resize_transform)
    
    eval_dataset = load_from_disk("/Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v1/validation").select(range(40)) 
    eval_dataset.set_transform(dynamic_resize_transform)
    # eval_dataset = None

    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir="/Projects/SG_VLN_HumanData/spatial_training/dump/pose_training_continue",
        # run_name="qwen-vln-action-dropout",
        save_strategy="steps",        # Save checkpoints frequently
        save_steps=100,          
                  # Save every 500 steps
        eval_strategy="steps",        # Check validation loss periodically
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
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=10000,   # We only need a few steps to hit peak memory
        report_to="tensorboard",
        # dataset_text_field="text",
        shuffle_dataset=True,

        assistant_only_loss=False,
        optim="paged_adamw_8bit",
        # dataloader_num_workers=8
        warmup_steps=12,
        remove_unused_columns=False,
        resume_from_checkpoint=resume_path
        # processing_class = processor
    )

    # 5. Initialize Trainer
    trainer = PoseTrainer(
        model=model,
        data_collator = PoseVLMCollator(processor=processor,max_length=TOTAL_BUDGET,dropout=0.6),
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
    trainer.train()
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