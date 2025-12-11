"""
Training utilities for SFT and DPO.
"""

import torch
import numpy as np
import random
from typing import Dict, Any, List, Optional
import os


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_training_args(args: Dict[str, Any], save_dir: str):
    """Save training arguments to file."""
    os.makedirs(save_dir, exist_ok=True)
    
    import json
    with open(os.path.join(save_dir, "training_args.json"), "w") as f:
        json.dump(args, f, indent=2, default=str)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    
    # Compute accuracy
    preds = np.argmax(predictions, axis=-1)
    mask = labels != -100
    correct = (preds == labels)[mask].sum()
    total = mask.sum()
    accuracy = correct / total if total > 0 else 0
    
    # Compute perplexity
    loss = -predictions[mask].mean()
    perplexity = np.exp(loss)
    
    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "loss": loss
    }


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for minimizing, "max" for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == "min":
            self.best_score = np.inf
            self.compare = lambda x, y: x < y - min_delta
        else:
            self.best_score = -np.inf
            self.compare = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float) -> bool:
        """Update early stopping state."""
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def gradient_accumulation_step(model, optimizer, loss, grad_accum_steps: int, grad_clip: float = 1.0):
    """Perform gradient accumulation step."""
    loss = loss / grad_accum_steps
    loss.backward()
    
    # Clip gradients
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    return loss.item() * grad_accum_steps


def prepare_alpaca_data(data_path: str, output_dir: str, tokenizer, max_examples: Optional[int] = None):
    """Prepare Alpaca data for training."""
    import json
    from sklearn.model_selection import train_test_split
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Limit examples if specified
    if max_examples is not None and max_examples < len(data):
        data = data[:max_examples]
    
    # Split into train/val
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "alpaca_train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, "alpaca_val.json"), "w") as f:
        json.dump(val_data, f, indent=2)
    
    # Create DPO data (simulated from Alpaca)
    dpo_train = []
    dpo_val = []
    
    for split_data, dpo_split in [(train_data, dpo_train), (val_data, dpo_val)]:
        for example in split_data:
            # Create a DPO example with chosen and rejected responses
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            chosen = example.get("output", "")
            
            # Create a "rejected" response (simulated)
            # In practice, you'd have real preference data
            if len(chosen) > 100:
                rejected = chosen[:len(chosen)//2] + "... [Truncated]"
            else:
                rejected = "I cannot provide a response to that instruction."
            
            dpo_example = {
                "instruction": instruction,
                "input": input_text,
                "chosen": chosen,
                "rejected": rejected
            }
            dpo_split.append(dpo_example)
    
    # Save DPO data
    with open(os.path.join(output_dir, "alpaca_dpo_train.json"), "w") as f:
        json.dump(dpo_train, f, indent=2)
    
    with open(os.path.join(output_dir, "alpaca_dpo_val.json"), "w") as f:
        json.dump(dpo_val, f, indent=2)
    
    print(f"Data prepared:")
    print(f"  SFT train: {len(train_data)} examples")
    print(f"  SFT val: {len(val_data)} examples")
    print(f"  DPO train: {len(dpo_train)} examples")
    print(f"  DPO val: {len(dpo_val)} examples")
    
    return train_data, val_data, dpo_train, dpo_val
