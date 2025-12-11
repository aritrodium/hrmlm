"""
Training script for HRMLM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import yaml
import os
import time
from tqdm import tqdm
from pathlib import Path

from model.hrmlm import HRMLM
from model.tokenizer import Tokenizer
from data.prepare_dataset import TextDataset
from utils.logging import setup_logging


class Trainer:
    """Main training class for HRMLM."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = setup_logging()
        
        # Initialize Accelerator for multi-GPU training
        self.accelerator = Accelerator(
            mixed_precision=self.config['optimization']['mixed_precision'],
            gradient_accumulation_steps=self.config['optimization']['gradient_accumulation_steps']
        )
        
        # Setup device
        self.device = self.accelerator.device
        
        # Initialize components
        self._init_tokenizer()
        self._init_datasets()
        self._init_model()
        self._init_optimizer()
        self._init_training()
        
    def _init_tokenizer(self):
        """Initialize tokenizer."""
        tokenizer_path = "models/tokenizer.model"
        if not os.path.exists(tokenizer_path):
            self.logger.info("Training new tokenizer...")
            tokenizer = Tokenizer()
            tokenizer.train(
                input_file=self.config['dataset']['train_file'],
                model_prefix="models/tokenizer",
                vocab_size=self.config['model']['vocab_size']
            )
        self.tokenizer = Tokenizer(tokenizer_path)
        
    def _init_datasets(self):
        """Initialize datasets and dataloaders."""
        self.logger.info("Loading datasets...")
        
        # Create datasets
        train_dataset = TextDataset(
            file_path=self.config['dataset']['train_file'],
            tokenizer=self.tokenizer,
            seq_len=self.config['model']['n_ctx']
        )
        
        val_dataset = TextDataset(
            file_path=self.config['dataset']['val_file'],
            tokenizer=self.tokenizer,
            seq_len=self.config['model']['n_ctx']
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['dataset']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=self.config['dataset']['pin_memory'],
            persistent_workers=self.config['dataset']['persistent_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['dataset']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
    def _init_model(self):
        """Initialize model."""
        self.logger.info("Initializing model...")
        
        self.model = HRMLM(
            vocab_size=self.tokenizer.vocab_size,
            n_ctx=self.config['model']['n_ctx'],
            n_embd=self.config['model']['n_embd'],
            n_hidden=self.config['model']['n_hidden'],
            n_high_hidden=self.config['model']['n_high_hidden'],
            T=self.config['model']['T'],
            N=self.config['model']['N'],
            dropout=self.config['model']['dropout'],
            layer_norm=self.config['model']['layer_norm']
        )
        
        # Compile model if enabled
        if self.config['optimization']['compile_model'] and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=tuple(self.config['training']['betas']),
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
    def _get_scheduler(self):
        """Create learning rate scheduler."""
        if self.config['training']['lr_scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['training']['warmup_steps']
            )
        elif self.config['training']['lr_scheduler'] == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=self.config['training']['warmup_steps']
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        
    def _init_training(self):
        """Initialize training components."""
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.config['checkpoint']['log_dir'])
        
        # Prepare with Accelerator
        self.model, self.optimizer, self.train_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
        
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                logits = self.model(input_ids)
                
                # Calculate loss
                loss = self.criterion(
                    logits.view(-1, self.tokenizer.vocab_size),
                    target_ids.view(-1)
                )
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                # Optimization step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                total_loss += loss.item()
                total_tokens += input_ids.numel()
                
                if batch_idx % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'lr': current_lr,
                        'ppl': torch.exp(loss).item()
                    })
                    
                    # TensorBoard logging
                    if self.accelerator.is_local_main_process:
                        global_step = epoch * len(self.train_loader) + batch_idx
                        self.writer.add_scalar('train/loss', loss.item(), global_step)
                        self.writer.add_scalar('train/lr', current_lr, global_step)
                        self.writer.add_scalar('train/ppl', torch.exp(loss).item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        
        for input_ids, target_ids in tqdm(self.val_loader, desc="Evaluating"):
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, self.tokenizer.vocab_size),
                target_ids.view(-1)
            )
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        if not self.accelerator.is_local_main_process:
            return
        
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
        self.accelerator.save_state(checkpoint_path)
        
        # Save configuration
        config_path = checkpoint_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean old checkpoints
        self._clean_old_checkpoints()
    
    def _clean_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoints = sorted(
            checkpoint_dir.glob("epoch_*.pth"),
            key=lambda x: int(x.stem.split('_')[1])
        )
        
        keep_last = self.config['checkpoint']['keep_last']
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            start_time = time.time()
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_loss = self.evaluate()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            
            # Log results
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch}: "
                           f"train_loss={train_loss:.4f}, "
                           f"val_loss={val_loss:.4f}, "
                           f"val_ppl={val_ppl:.2f}, "
                           f"time={epoch_time:.2f}s")
            
            # TensorBoard logging
            if self.accelerator.is_local_main_process:
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/ppl', val_ppl, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = Path(self.config['checkpoint']['save_dir']) / "best_model.pth"
                self.accelerator.save_state(best_path)
        
        self.logger.info("Training completed!")
        self.writer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HRMLM")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = Trainer(args.config)
    trainer.train()
