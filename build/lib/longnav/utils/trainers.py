import torch
from torch import nn
from trl import SFTTrainer
from transformers.trainer_utils import EvalPrediction
from peft import PeftType

# Assuming this helper exists in your utils, otherwise define it:
def entropy_from_logits(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)

class PrunedSFTTrainer(SFTTrainer):
    
    def _prune_tensor(self, tensor, mask, padding_value=0):
        """
        Slices a tensor (B, L) using a boolean mask (B, L) 
        and re-pads it to (B, L_new) to match the pruned logits structure.
        """
        if tensor is None:
            return None
        return tensor[:,mask]
  
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        """
        Compute training loss and additionally compute token accuracies
        Adjusted to handle Visual Pruning mismatch between Inputs and Outputs.
        """
        mode = "train" if self.model.training else "eval"

        # 1. PREPARE INPUTS (Standard TRL logic)
        labels = inputs["labels"] if "shift_labels" not in inputs else None
        inputs["use_cache"] = False
        
        if self.args.use_liger_kernel:
            inputs["return_token_accuracy"] = True
            inputs["use_token_scaling"] = self.args.loss_type == "dft"

        # 2. RUN FORWARD PASS (Super handles the model call)
        # The model internally prunes and computes the correct loss.
        # (loss, outputs) = super(SFTTrainer,self).compute_loss(
        #     model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        # )
        outputs = model(**inputs)
        loss = outputs['loss']
        # =========================================================================
        # CRITICAL FIX: SYNC INPUTS WITH PRUNED OUTPUTS
        # =========================================================================
        # We need the pruning mask to align inputs['labels'] and inputs['attention_mask']
        # to the pruned outputs.logits shape.
        
        # Check if model returned the mask (Your model MUST return this)
        seq_keep_mask = getattr(outputs, "seq_keep_mask", None)

        if seq_keep_mask is not None:
            # 1. Update Attention Mask (used for Entropy and Token Counting)
            # We slice it so metrics don't count dropped visual tokens.
            if "attention_mask" in inputs:
                inputs["attention_mask"] = self._prune_tensor(
                    inputs["attention_mask"], seq_keep_mask, padding_value=0
                )
            
            # 2. Update Labels (used for Accuracy)
            # -100 is standard ignore index
            if "labels" in inputs:
                inputs["labels"] = self._prune_tensor(
                    inputs["labels"], seq_keep_mask, padding_value=-100
                )
                labels = inputs["labels"] # Update local reference

            if "shift_labels" in inputs:
                 inputs["shift_labels"] = self._prune_tensor(
                    inputs["shift_labels"], seq_keep_mask, padding_value=-100
                )
        else:
            print("ERROR: no sqm")
            exit()
        # =========================================================================


        # 3. METRICS LOGIC (Now safely using pruned inputs)

        # # Compute entropy
        # if not self.args.use_liger_kernel: 
        #     with torch.no_grad():
        #         per_token_entropy = entropy_from_logits(outputs.logits)
                
        #         if (
        #             self.num_virtual_tokens > 0
        #             and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
        #         ):
        #             per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
                
        #         if "attention_mask" in inputs:
        #             # Uses the PRUNED attention_mask now
        #             attention_mask = inputs["attention_mask"]
        #             # Ensure shapes match (handle edge cases where padding adds 1)
        #             min_len = min(attention_mask.shape[1], per_token_entropy.shape[1])
        #             attention_mask = attention_mask[:, :min_len]
        #             per_token_entropy = per_token_entropy[:, :min_len]
                    
        #             entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        #         elif "position_ids" in inputs:
        #             entropy = torch.mean(per_token_entropy)
        #         else:
        #             raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                
        #         entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        #     self._metrics[mode]["entropy"].append(entropy)

        # # Compute Token Counts (for tokens/sec)
        # if mode == "train":
        #     if "attention_mask" in inputs:
        #         # Uses PRUNED mask -> Correctly counts only processed tokens
        #         num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
        #     elif "position_ids" in inputs:
        #         # Fallback (Might be inaccurate if position_ids weren't pruned, but less likely path)
        #         local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
        #         num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
        #     else:
        #         raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        #     self._total_train_tokens += num_tokens_in_batch
        # self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute Accuracy
        if self.args.use_liger_kernel:
            # Liger handles its own accuracy internally, assuming it received pruned inputs inside forward
            token_accuracy = self.accelerator.gather_for_metrics(outputs.token_accuracy).mean().item()
            self._metrics[mode]["mean_token_accuracy"].append(token_accuracy)
        else:
            with torch.no_grad():
                if "shift_labels" in inputs:
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Shape Safety Check before argmax
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100

                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()

                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)

                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # if self.aux_loss_enabled:
        #     aux_loss = outputs.aux_loss
        #     aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
        #     self._metrics[mode]["aux_loss"].append(aux_loss)

        return (loss, outputs) if return_outputs else loss