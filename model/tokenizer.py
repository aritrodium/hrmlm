"""
SentencePiece tokenizer wrapper with chat templates.
"""

import sentencepiece as spm
import os
import json
from typing import List, Optional, Dict, Any
import warnings


class Tokenizer:
    """Wrapper for SentencePiece tokenizer with chat templates."""
    
    def __init__(self, model_path: str):
        """
        Initialize tokenizer from SentencePiece model.
        
        Args:
            model_path: Path to SentencePiece model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Special tokens
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        
        # Add special tokens for instruction following
        self.instruction_token = "<|im_start|>"
        self.response_token = "<|im_end|>"
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        
        # Add these to vocabulary
        self.special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "instruction_token": self.instruction_token,
            "response_token": self.response_token,
            "system_token": self.system_token,
            "user_token": self.user_token,
            "assistant_token": self.assistant_token,
        }
        
        # Vocabulary
        self.vocab_size = self.sp.get_piece_size()
        
        # Chat template
        self.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"""
    
    def encode(self, 
               text: str, 
               add_bos: bool = False,
               add_eos: bool = False,
               max_length: Optional[int] = None,
               truncation: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.sp.encode(text, out_type=int)
        
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
            
        if max_length is not None and truncation:
            tokens = tokens[:max_length]
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            # Remove special tokens
            token_ids = [tid for tid in token_ids 
                        if tid not in [self.pad_token_id, self.unk_token_id, 
                                      self.bos_token_id, self.eos_token_id]]
        
        # Decode with SentencePiece
        text = self.sp.decode(token_ids)
        
        # Remove special tokens from text if they weren't removed earlier
        if skip_special_tokens:
            for token in [self.instruction_token, self.response_token, 
                         self.system_token, self.user_token, self.assistant_token]:
                text = text.replace(token, "")
        
        return text
    
    def encode_instruction(self, 
                          instruction: str,
                          input_text: Optional[str] = None,
                          response: Optional[str] = None,
                          max_length: int = 1024) -> Dict[str, Any]:
        """
        Encode an instruction for SFT training.
        
        Args:
            instruction: The instruction text
            input_text: Optional input context
            response: Optional response for training
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Build prompt
        prompt = f"{self.instruction_token}instruction\n{instruction}\n{self.response_token}"
        if input_text:
            prompt += f"\n{self.instruction_token}input\n{input_text}\n{self.response_token}"
        prompt += f"\n{self.instruction_token}response\n"
        
        # Encode prompt
        prompt_ids = self.encode(prompt, add_bos=True, add_eos=False)
        
        if response:
            # Training mode: include response
            response_ids = self.encode(response, add_bos=False, add_eos=True)
            input_ids = prompt_ids + response_ids
            
            # Create labels: -100 for prompt, actual tokens for response
            labels = [-100] * len(prompt_ids) + response_ids
            
            # Truncate if necessary
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            
            # Create attention mask
            attention_mask = [1] * len(input_ids)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # Inference mode: just prompt
            if len(prompt_ids) > max_length:
                prompt_ids = prompt_ids[:max_length]
            
            attention_mask = [1] * len(prompt_ids)
            
            return {
                "input_ids": prompt_ids,
                "attention_mask": attention_mask
            }
    
    def apply_chat_template(self, 
                           messages: List[Dict[str, str]],
                           add_generation_prompt: bool = False) -> str:
        """
        Apply chat template to messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: Whether to add assistant prompt at the end
            
        Returns:
            Formatted chat string
        """
        from jinja2 import Template
        
        template = Template(self.chat_template)
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt
        )
    
    def train(self, 
              input_file: str, 
              model_prefix: str,
              vocab_size: int = 32000,
              model_type: str = "bpe",
              character_coverage: float = 1.0):
        """
        Train a new SentencePiece tokenizer.
        
        Args:
            input_file: Path to text file for training
            model_prefix: Output model prefix
            vocab_size: Size of vocabulary
            model_type: Tokenization algorithm (bpe, unigram, char, word)
            character_coverage: Character coverage ratio
        """
        # Create training arguments
        train_args = [
            f"--input={input_file}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={model_type}",
            f"--character_coverage={character_coverage}",
            "--pad_id=0",
            "--unk_id=1",
            "--bos_id=2",
            "--eos_id=3",
            "--max_sentence_length=16384",
            "--add_dummy_prefix=false",
            "--remove_extra_whitespaces=true",
            "--normalization_rule_name=nmt_nfkc"
        ]
        
        # Train tokenizer
        spm.SentencePieceTrainer.train(" ".join(train_args))
        
        # Save special tokens
        special_tokens_path = f"{model_prefix}_special_tokens.json"
        with open(special_tokens_path, "w") as f:
            json.dump(self.special_tokens, f, indent=2)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save SentencePiece model
        model_path = os.path.join(save_directory, "tokenizer.model")
        
        # Copy the model file
        import shutil
        shutil.copy2(self.sp.get_serialized_proto().decode('utf-8'), model_path)
        
        # Save special tokens and config
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "chat_template": self.chat_template
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
