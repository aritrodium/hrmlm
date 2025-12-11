"""
Supervised Fine-Tuning (SFT) script for HRMLM on Alpaca dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import os
import json
from tqdm import tqdm
import wandb
from datetime import datetime

from model.hrmlm import HRMLM
from model.tokenizer import Tokenizer
from data.alpaca_dataset import load_alpaca_dataset
from utils.logging import setup_logging


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = setup_logging()
        
        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config['training'].get('use_cpu', False) 
            else "cpu"
        )
        
        # Initialize components
        self._init_tokenizer()
        self._init_model()
        self._init_datasets()
        self._init_optimizer()
        
        # Setup Weights & Biases
        if self.config['logging'].get('use_wandb', False):
            wandb.init(
                project=self.config['logging'].get('wandb_project', 'hrmlm-sft'),
                name=self.config['logging'].get('run_name', f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                config=self.config
            )
    
    def _init_tokenizer(self):
        """Initialize tokenizer."""
        tokenizer_path = self.config['tokenizer']['model_path']
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        self.tokenizer = Tokenizer(tokenizer_path)
        self.logger.info(f"Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
    
    def _init_model(self):
        """Initialize model from pretrained checkpoint."""
        model_config = self.config['model']
        
        # Create model
        self.model = HRMLM(
            vocab_size=self.tokenizer.vocab_size,
            n_ctx=model_config.get('n_ctx', 1024),
            n_embd=model_config.get('n_embd', 768),
            n_hidden=model_config.get('n_hidden', 1024),
            n_high_hidden=model_config.get('n_high_hidden', 768),
            T=model_config.get('T', 5),
            N=model_config.get('N', 3),
            dropout=model_config.get('dropout', 0.1),
            layer_norm=model_config.get('layer_norm', True)
        )
        
        # Load pretrained weights if specified
        pretrained_path = self.config['training'].get('pretrained_path')
        if pretrained_path and os.path.exists(pretrained_path):
            self.logger.info(f"Loading pretrained weights from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        else:
            self.logger.warning("No pretrained weights found, training from scratch")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Print model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _init_datasets(self):
        """Initialize datasets and dataloaders."""
        self.logger.info("Loading datasets...")
        
        # Training dataset
        self.train_loader = load_alpaca_dataset(
            data_path=self.config['dataset']['train_path'],
            tokenizer=self.tokenizer,
            split="train",
            batch_size=self.config['training']['batch_size'],
            max_length=self.config['dataset']['max_length'],
            is_dpo=False,
            num_examples=self.config['dataset'].get('num_train_examples'),
            shuffle=True
        )
        
        # Validation dataset
        self.val_loader = load_alpaca_dataset(
            data_path=self.config['dataset']['val_path'],
            tokenizer=self.tokenizer,
            split="val",
            batch_size=self.config['training']['eval_batch_size'],
            max_length=self.config['dataset']['max_length'],
            is_dpo=False,
            num_examples=self.config['dataset'].get('num_val_examples'),
            shuffle=False
        )
        
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        training_config = self.config['training']
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            betas=tuple(training_config.get('betas', [0.9, 0.95])),
            weight_decay=training_config.get('weight_decay', 0.1),
            eps=training_config.get('eps', 1e-8)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * training_config['epochs']
        warmup_steps = training_config.get('warmup_steps', 0)
        
        if training_config.get('scheduler', 'cosine') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        
        # Gradient clipping
        self.grad_clip = training_config.get('grad_clip', 1.0)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update statistics
            total_loss += outputs.loss.item() * self.gradient_accumulation_steps
            total_tokens = batch['attention_mask'].sum().item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': outputs.loss.item(),
                'lr': current_lr,
                'ppl': torch.exp(outputs.loss).item()
            })
            
            # Log to wandb
            if self.config['logging'].get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': outputs.loss.item(),
                    'train/ppl': torch.exp(outputs.loss).item(),
                    'train/lr': current_lr,
                    'train/step': epoch * len(self.train_loader) + batch_idx
                })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, epoch: int = 0):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            total_loss += outputs.loss.item()
            total_tokens += batch['attention_mask'].sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        # Log to wandb
        if self.config['logging'].get('use_wandb', False):
            wandb.log({
                'val/loss': avg_loss,
                'val/ppl': avg_ppl,
                'val/epoch': epoch
            })
        
        return avg_loss, avg_ppl
    
    @torch.no_grad()
    def generate_examples(self, num_examples: int = 3):
        """Generate example responses for evaluation."""
        self.model.eval()
        
        # Sample prompts from validation set
        examples = []
        for batch in self.val_loader:
            for i in range(min(num_examples, batch['input_ids'].size(0))):
                # Extract prompt (up to first -100 in labels)
                input_ids = batch['input_ids'][i]
                labels = batch['labels'][i]
                
                # Find where response starts (first non -100)
                response_start = None
                for j in range(len(labels)):
                    if labels[j] != -100:
                        response_start = j
                        break
                
                if response_start is None:
                    continue
                
                prompt_ids = input_ids[:response_start]
                
                # Generate response
                generated = self.model.generate(
                    input_ids=prompt_ids.unsqueeze(0).to(self.device),
                    max_length=self.config['generation']['max_length'],
                    temperature=self.config['generation']['temperature'],
                    top_k=self.config['generation']['top_k'],
                    top_p=self.config['generation']['top_p'],
                    do_sample=True
                )
                
                # Decode
                prompt = self.tokenizer.decode(prompt_ids.tolist())
                generated_text = self.tokenizer.decode(generated[0].tolist())
                
                examples.append({
                    'prompt': prompt,
                    'generated': generated_text
                })
                
                if len(examples) >= num_examples:
                    break
            
            if len(examples) >= num_examples:
                break
        
        return examples
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config['checkpoint']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        checkpoint_path = os.path.join(checkpoint_dir, f"sft_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, "sft_best.pth")
            torch.save(self.model.state_dict(), best_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean old checkpoints
        self._clean_old_checkpoints()
    
    def _clean_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoint_dir = self.config['checkpoint']['save_dir']
        keep_last = self.config['checkpoint'].get('keep_last', 3)
        
        if not os.path.exists(checkpoint_dir):
            return
        
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith("sft_epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.split('_')[2].split('.')[0])
        )
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                os.remove(os.path.join(checkpoint_dir, checkpoint))
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting SFT training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config['training']['epochs']}")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch}/{self.config['training']['epochs']}")
            self.logger.info(f"{'='*50}")
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, val_ppl = self.evaluate(epoch)
            
            # Log results
            self.logger.info(f"Epoch {epoch} Results:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}")
            self.logger.info(f"  Val PPL: {val_ppl:.2f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.logger.info(f"  New best model! Val loss: {val_loss:.4f}")
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Generate examples
            if epoch % self.config['generation'].get('example_freq', 1) == 0:
                self.logger.info("\nGenerating examples:")
                examples = self.generate_examples(num_examples=2)
                for i, example in enumerate(examples):
                    self.logger.info(f"\nExample {i+1}:")
                    self.logger.info(f"  Prompt: {example['prompt'][:200]}...")
                    self.logger.info(f"  Generated: {example['generated'][:200]}...")
        
        self.logger.info("\nSFT training completed!")
        
        # Save final model
        final_path = os.path.join(self.config['checkpoint']['save_dir'], "sft_final.pth")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model saved: {final_path}")
        
        # Close wandb
        if self.config['logging'].get('use_wandb', False):
            wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HRMLM with SFT")
    parser.add_argument("--config", type=str, default="config_sft.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = SFTTrainer(args.config)
    trainer.train()
