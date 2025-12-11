"""
Alpaca dataset loader for SFT and DPO training.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import random
from model.tokenizer import Tokenizer
import numpy as np


class AlpacaDataset(Dataset):
    """
    Dataset for Alpaca instruction following.
    Supports both SFT and DPO data formats.
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: Tokenizer,
                 max_length: int = 1024,
                 split: str = "train",
                 is_dpo: bool = False,
                 num_examples: Optional[int] = None):
        """
        Initialize Alpaca dataset.
        
        Args:
            data_path: Path to Alpaca JSON file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            split: Dataset split ("train", "val", "test")
            is_dpo: Whether to load DPO format data
            num_examples: Number of examples to use (None = all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_dpo = is_dpo
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter by split if specified
        if "split" in self.data[0]:
            self.data = [d for d in self.data if d.get("split", "train") == split]
        
        # Limit number of examples
        if num_examples is not None:
            self.data = self.data[:num_examples]
        
        # For DPO, we need to create preference pairs
        if is_dpo:
            self._prepare_dpo_data()
    
    def _prepare_dpo_data(self):
        """Prepare DPO data from Alpaca format."""
        dpo_data = []
        
        for example in self.data:
            # Create a "chosen" response (the actual instruction output)
            chosen_example = example.copy()
            
            # Create a "rejected" response
            # In practice, you'd have actual rejected responses
            # Here we simulate by modifying the instruction or using a weaker response
            rejected_example = example.copy()
            
            # Simple heuristic: make the rejected response shorter or less informative
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            
            if len(output) > 50:
                # Truncate for rejected response
                rejected_output = output[:len(output)//2] + "... [Response truncated]"
            else:
                # Add a generic weak response
                rejected_output = "I'm not sure how to respond to that instruction."
            
            rejected_example["output"] = rejected_output
            
            dpo_data.append({
                "instruction": instruction,
                "input": example.get("input", ""),
                "chosen": output,
                "rejected": rejected_output
            })
        
        self.data = dpo_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        if self.is_dpo:
            return self._get_dpo_example(example)
        else:
            return self._get_sft_example(example)
    
    def _get_sft_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        """Get example for SFT training."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        # Encode instruction
        encoded = self.tokenizer.encode_instruction(
            instruction=instruction,
            input_text=input_text if input_text else None,
            response=output,
            max_length=self.max_length
        )
        
        # Convert to tensors
        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        }
        
        if "labels" in encoded:
            item["labels"] = torch.tensor(encoded["labels"], dtype=torch.long)
        
        return item
    
    def _get_dpo_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        """Get example for DPO training."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        chosen_response = example.get("chosen", "")
        rejected_response = example.get("rejected", "")
        
        # Encode chosen response
        chosen_encoded = self.tokenizer.encode_instruction(
            instruction=instruction,
            input_text=input_text if input_text else None,
            response=chosen_response,
            max_length=self.max_length
        )
        
        # Encode rejected response
        rejected_encoded = self.tokenizer.encode_instruction(
            instruction=instruction,
            input_text=input_text if input_text else None,
            response=rejected_response,
            max_length=self.max_length
        )
        
        return {
            "chosen_input_ids": torch.tensor(chosen_encoded["input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_encoded["attention_mask"], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_encoded.get("labels", chosen_encoded["input_ids"]), dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_encoded["input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_encoded["attention_mask"], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_encoded.get("labels", rejected_encoded["input_ids"]), dtype=torch.long),
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict], is_dpo: bool = False) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        if is_dpo:
            return AlpacaDataset._collate_dpo_batch(batch)
        else:
            return AlpacaDataset._collate_sft_batch(batch)
    
    @staticmethod
    def _collate_sft_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate SFT batch."""
        # Find max length in batch
        max_len = max(len(item["input_ids"]) for item in batch)
        
        # Initialize tensors
        batch_size = len(batch)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = item["input_ids"]
            attention_mask[i, :seq_len] = item["attention_mask"]
            if "labels" in item:
                labels[i, :seq_len] = item["labels"]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    @staticmethod
    def _collate_dpo_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate DPO batch."""
        # Find max lengths
        max_chosen_len = max(len(item["chosen_input_ids"]) for item in batch)
        max_rejected_len = max(len(item["rejected_input_ids"]) for item in batch)
        max_len = max(max_chosen_len, max_rejected_len)
        
        batch_size = len(batch)
        
        # Initialize tensors for chosen
        chosen_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        chosen_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        chosen_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        # Initialize tensors for rejected
        rejected_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        rejected_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        rejected_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            # Chosen
            chosen_len = len(item["chosen_input_ids"])
            chosen_input_ids[i, :chosen_len] = item["chosen_input_ids"]
            chosen_attention_mask[i, :chosen_len] = item["chosen_attention_mask"]
            chosen_labels[i, :chosen_len] = item["chosen_labels"]
            
            # Rejected
            rejected_len = len(item["rejected_input_ids"])
            rejected_input_ids[i, :rejected_len] = item["rejected_input_ids"]
            rejected_attention_mask[i, :rejected_len] = item["rejected_attention_mask"]
            rejected_labels[i, :rejected_len] = item["rejected_labels"]
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }


def load_alpaca_dataset(data_path: str, 
                        tokenizer: Tokenizer,
                        split: str = "train",
                        batch_size: int = 4,
                        max_length: int = 1024,
                        is_dpo: bool = False,
                        num_examples: Optional[int] = None,
                        shuffle: bool = True):
    """
    Load Alpaca dataset with DataLoader.
    
    Args:
        data_path: Path to Alpaca JSON
        tokenizer: Tokenizer instance
        split: Dataset split
        batch_size: Batch size
        max_length: Maximum sequence length
        is_dpo: Whether to load DPO format
        num_examples: Number of examples to use
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader for Alpaca dataset
    """
    from torch.utils.data import DataLoader
    
    dataset = AlpacaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        is_dpo=is_dpo,
        num_examples=num_examples
    )
    
    collate_fn = lambda batch: AlpacaDataset.collate_fn(batch, is_dpo=is_dpo)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader
