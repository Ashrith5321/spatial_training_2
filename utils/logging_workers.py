from typing import Any, Dict
import ray
import wandb
import numpy as np
import os
import sys
import subprocess

class WandbLoggerActor:
    def __init__(self, wandb_init_kwargs, run_config=None, log_raw=False, commit_interval=5):
        """
        Args:
            wandb_init_kwargs: Dict for wandb.init (project, entity, name).
            run_config: The main experiment config dict (hyperparameters).
            log_raw: Boolean toggle for raw data in tables.
            commit_interval: Frequency of table uploads.
        """
        # 1. Gather Environment Metadata (SLURM, Git)
        system_metadata = self._capture_system_metadata()
        
        # 2. Merge User Config with System Metadata
        # This ensures we don't overwrite user config, but append system info
        full_config = run_config if run_config else {}
        full_config.update(system_metadata)

        # 3. Initialize WandB
        # We pass the merged config here so it appears in the "Overview" tab
        self.run = wandb.init(
            **wandb_init_kwargs, 
            config=full_config, 
            reinit=True
        )
        
        self.log_raw = log_raw
        self.commit_interval = commit_interval
        
        # Table State
        self.table = None
        self.columns = None
        self.rows_since_last_commit = 0

        self.defined_metrics = set()

    def log_global_metrics(self, metrics: dict, step=None):
        """
        For Driver-side metrics: Training Loss, Learning Rate, Epoch, etc.
        """
        if step is not None:
            # If step is provided, we align the metric to that X-axis
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)
    def log(self,row):
        """
        Processes a single episode row with namespace-based media detection.
        """
        # --- 4. Log Live Scalars ---
        # Filter out heavy objects (Lists, Arrays, and the new Media objects)
        # This keeps the "Run Overview" page clean and fast.
        log_payload = {
            k: v for k, v in row.items() 
            if not isinstance(v, (list, np.ndarray, wandb.Image, wandb.Video))
        }
        for k, v in log_payload.items():
            if k not in self.defined_metrics:
                # Filter for numeric types (exclude bools, strings, and None)
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    self.run.define_metric(k, summary="mean")
                    self.run.define_metric(k, summary="max")
                    self.run.define_metric(k, summary="min")
                    self.defined_metrics.add(k)
        self.run.log(log_payload)

    def log_row(self, row:Dict[str,Any]):
        """
        Processes a single episode row with namespace-based media detection.
        """
        processed_row = {}
        
        # --- 1. Process & Filter ---
        for k, v in row.items():
            # A. Raw Data Toggle
            if k.startswith('raw/') and not self.log_raw:
                continue
            
            # B. Dynamic Media Namespaces
            if k.startswith('img/') and v is not None:
                # Wraps numpy arrays or paths into WandB Images
                processed_row[k] = wandb.Image(v)
            elif k.startswith('vid/') and v is not None:
                # Wraps paths into WandB Videos (expects shared filesystem)
                processed_row[k] = wandb.Video(v, format="mp4")
            else:
                # C. Pass-through (Scalars, Strings, Lists, or None values)
                processed_row[k] = v
                 # --- New: Accumulate Summary Stats ---
                # Check if the value is a scalar (int, float, or numpy number)
        # --- 2. Lazy Table Init ---
        if self.table is None:
            self.columns = sorted(list(processed_row.keys()))
            self.table = wandb.Table(columns=self.columns,allow_mixed_types=True,log_mode='MUTABLE')

        # --- 3. Add to Buffer ---
        # Use .get() to handle potential schema drift (though rare in this setup)
        table_row = [processed_row.get(c, None) for c in self.columns]
        self.table.add_data(*table_row)
        self.rows_since_last_commit += 1

        # --- 4. Log Live Scalars ---
        # Filter out heavy objects (Lists, Arrays, and the new Media objects)
        # This keeps the "Run Overview" page clean and fast.
        log_payload = {
            k: v for k, v in processed_row.items() 
            if not isinstance(v, (list, np.ndarray, wandb.Image, wandb.Video))
        }
        for k, v in log_payload.items():
            if k not in self.defined_metrics:
                # Filter for numeric types (exclude bools, strings, and None)
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    self.run.define_metric(k, summary="mean")
                    self.run.define_metric(k, summary="max")
                    self.run.define_metric(k, summary="min")
                    self.defined_metrics.add(k)

        # --- 5. Conditional Commit ---
        if self.rows_since_last_commit >= self.commit_interval:
            log_payload["episode_details"] = self.table
            # Reset table to prevent O(N^2) slowdown and race conditions
            # self.table = wandb.Table(columns=self.columns, allow_mixed_types=True)
            self.rows_since_last_commit = 0 
            
        self.run.log(log_payload)

    def _capture_system_metadata(self):
        """
        Internal helper to grab SLURM and Git info.
        """
        meta = {}
        
        # A. SLURM Environment Variables
        # These are standard across most SLURM clusters
        slurm_keys = [
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
            "SLURM_NTASKS", "SLURM_CPUS_PER_TASK"
        ]
        for k in slurm_keys:
            if k in os.environ:
                meta[f"system/{k.lower()}"] = os.environ[k]

        # B. Git Commit Hash (Reproducibility)
        # We try to run git rev-parse; if it fails (not a repo), we skip.
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).strip().decode('utf-8')
            meta["system/git_commit"] = commit_hash
        except:
            pass

        return meta

    def alert(self, 
        title: str,
        text: str,
        level):
        """
        Error logging.
        Args:
            message: The error string.
            level: ERROR, INFO, WARN
            alert: If True, triggers a WandB Alert (useful for cluster failures).
        """       
        self.run.alert(
            title=title, 
            text=text, 
            level=level
        )  
        
    def flush(self):
        """Forces a commit of the current table buffer to WandB."""
        if self.rows_since_last_commit > 0 and self.table is not None:
            # Create a payload with just the table
            payload = {"episode_details": self.table}
            self.run.log(payload)
            
            # Reset the counter
            self.rows_since_last_commit = 0
    def close(self):
        # Flush the table one last time
        if self.table and self.rows_since_last_commit > 0:
            self.run.log({"episode_details": self.table})
        self.run.finish()