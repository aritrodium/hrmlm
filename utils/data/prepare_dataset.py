"""
Dataset preparation utilities.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
from pathlib import Path


class TextDataset(Dataset):
    """Dataset for text language modeling with sliding window."""
    
    def __init__(self, 
                 file_path: str, 
                 tokenizer,
                 seq_len: int = 1024,
                 stride: int = 512):
        """
        Initialize dataset.
        
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer instance
            seq_len: Sequence length
            stride: Stride for sliding window (overlap = seq_len - stride)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        
        # Load and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        self._create_chunks()
    
    def _create_chunks(self):
        """Create overlapping chunks of tokens."""
        self.chunks = []
        for i in range(0, len(self.tokens) - self.seq_len, self.stride):
            chunk = self.tokens[i:i + self.seq_len]
            if len(chunk) == self.seq_len:
                self.chunks.append(chunk)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids


def split_dataset(input_file: str, 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1):
    """
    Split text file into train/val/test sets.
    
    Args:
        input_file: Path to input text file
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle lines
    np.random.seed(42)
    np.random.shuffle(lines)
    
    # Split indices
    n_total = len(lines)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Create splits
    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]
    
    # Save splits
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.txt", 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(output_dir / "val.txt", 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    with open(output_dir / "test.txt", 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_lines)} lines")
    print(f"  Val: {len(val_lines)} lines")
    print(f"  Test: {len(test_lines)} lines")
