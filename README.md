# HRMLM: Hierarchical Recurrent Memory Language Model 


## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Inference](#inference)
8. [API Reference](#api-reference)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

## Overview

**HRMLM** (Hierarchical Recurrent Memory Language Model) is a PyTorch implementation of a novel neural architecture that combines hierarchical recurrence with multi-timescale processing for language modeling. The model features a two-level RNN structure where low-level and high-level components operate at different temporal resolutions, enabling better capture of both local dependencies and global context.

### Key Features

- 🏗️ **Hierarchical Architecture**: Dual-level RNN with configurable temporal cycles
- ⚡ **Efficient Training**: Support for gradient accumulation, mixed precision, and multi-GPU
- 🔧 **Professional Configuration**: YAML-based configuration with industry-standard parameters
- 📊 **Comprehensive Logging**: TensorBoard integration and structured logging
- 💾 **Checkpoint Management**: Automatic checkpointing with resume capability
- 🎯 **Text Generation**: Built-in sampling methods (top-k, top-p, temperature)

## Architecture

### Model Structure

```
Input
  ↓
Embedding Layer (n_embd dimensions)
  ↓
Hierarchical Processing:
  ┌─────────────────────────────────────┐
  │ For each token:                     │
  │   For N high-level cycles:          │
  │     For T low-level cycles:         │
  │       Low-Level RNN (fast timescale)│
  │     High-Level RNN (slow timescale) │
  └─────────────────────────────────────┘
  ↓
Output Projection
  ↓
Vocabulary Distribution
```

### Mathematical Formulation

The HRMLM processes sequences through a hierarchical recurrence:

1. **Low-level RNN** (fast timescale, T cycles):
   ```
   h_l^{(t,n,τ+1)} = GRU_low([x^{(t)}; h_h^{(t,n)}], h_l^{(t,n,τ)})
   ```
   Where:
   - `t`: token position
   - `n`: high-level cycle index
   - `τ`: low-level cycle index (0 ≤ τ < T)

2. **High-level RNN** (slow timescale, N cycles):
   ```
   h_h^{(t,n+1)} = GRU_high(h_l^{(t,n,T)}, h_h^{(t,n)})
   ```

3. **Output** for position t:
   ```
   y^{(t)} = softmax(W_o · h_h^{(t,N)} + b_o)
   ```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.3+ (for GPU training)
- 16GB+ RAM (for training large models)

### Installation Methods

#### Method 1: Pip Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hrmlm.git
cd hrmlm

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

#### Method 2: Conda Installation

```bash
# Create conda environment
conda create -n hrmlm python=3.9
conda activate hrmlm

# Install PyTorch (choose appropriate version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

#### Method 3: Docker

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py", "--config", "config.yaml"]
```

## Usage

### Quick Start

```python
from model.hrmlm import HRMLM
from model.tokenizer import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer("models/tokenizer.model")

# Initialize model
model = HRMLM(
    vocab_size=tokenizer.vocab_size,
    n_ctx=1024,
    n_embd=768,
    n_hidden=1024,
    n_high_hidden=768,
    T=5,
    N=3,
    dropout=0.1
)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train or load pretrained weights
model.load_state_dict(torch.load("checkpoints/best_model.pth"))

# Generate text
generated = model.generate(
    prompt_ids=tokenizer.encode("Once upon a time"),
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
print(tokenizer.decode(generated[0]))
```

### Command Line Interface

#### Training

```bash
# Basic training
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/epoch_3.pth

# Train with specific GPU
CUDA_VISIBLE_DEVICES=0,1 python train.py --config config.yaml

# Distributed training
accelerate launch --num_processes 4 train.py --config config.yaml
```

#### Text Generation

```bash
# Interactive generation
python generate.py --checkpoint checkpoints/best_model.pth

# Batch generation from file
python generate.py \
    --checkpoint checkpoints/best_model.pth \
    --input_file prompts.txt \
    --output_file generations.txt \
    --max_length 200 \
    --temperature 0.7

# Server mode
python server.py --checkpoint checkpoints/best_model.pth --port 8000
```

## Configuration

### Configuration File (`config.yaml`)

```yaml
# Model Configuration
model:
  # Architecture
  architecture: "hrmlm"           # Model type
  vocab_size: 32000               # Vocabulary size (must match tokenizer)
  
  # Dimensions
  n_ctx: 1024                     # Context window size
  n_embd: 768                     # Embedding dimension
  n_hidden: 1024                  # Low-level hidden dimension
  n_high_hidden: 768              # High-level hidden dimension
  
  # Temporal parameters
  T: 5                            # Low-level cycles per step
  N: 3                            # High-level cycles per token
  
  # Regularization
  dropout: 0.1                    # Dropout rate
  layer_norm: true                # Use layer normalization

# Tokenizer Configuration
tokenizer:
  model_type: "bpe"               # "bpe", "unigram", "char", or "word"
  vocab_size: 32000               # Must match model.vocab_size
  character_coverage: 1.0         # Character coverage for SentencePiece
  max_sentence_length: 16384      # Maximum sentence length
  add_dummy_prefix: false         # Add dummy prefix for consistency
  remove_extra_whitespaces: true  # Remove extra whitespaces

# Dataset Configuration
dataset:
  # File paths
  train_file: "data/processed/train.txt"
  val_file: "data/processed/val.txt"
  test_file: "data/processed/test.txt"
  
  # Processing
  sequence_length: 1024           # Must match model.n_ctx
  batch_size: 32                  # Batch size per device
  
  # DataLoader settings
  num_workers: 4                  # Data loading workers
  pin_memory: true                # Pin memory for faster transfer
  persistent_workers: true        # Keep workers alive
  shuffle: true                   # Shuffle training data
  
  # Dataset splitting
  train_split: 0.8                # Training set proportion
  val_split: 0.1                  # Validation set proportion
  test_split: 0.1                 # Test set proportion

# Training Configuration
training:
  # Schedule
  epochs: 10                      # Number of training epochs
  max_steps: 100000               # Maximum training steps (optional)
  
  # Optimization
  learning_rate: 6e-4             # Base learning rate
  betas: [0.9, 0.95]              # Adam beta parameters
  weight_decay: 0.1               # L2 regularization
  grad_clip: 1.0                  # Gradient clipping norm
  
  # Learning rate schedule
  lr_scheduler: "cosine"          # "cosine", "linear", or "constant"
  warmup_steps: 2000              # Learning rate warmup steps
  min_lr: 1e-5                    # Minimum learning rate
  
  # Evaluation
  eval_steps: 500                 # Steps between evaluations
  eval_batch_size: 16             # Batch size for evaluation
  save_steps: 1000                # Steps between checkpoints

# Optimization
optimization:
  gradient_accumulation_steps: 4   # Steps for gradient accumulation
  mixed_precision: "fp16"          # "no", "fp16", or "bf16"
  compile_model: true              # Use torch.compile (PyTorch 2.0+)
  use_flash_attention: false       # Use flash attention (if available)

# Checkpointing
checkpoint:
  save_dir: "checkpoints"         # Directory for checkpoints
  save_every: 1000                # Save checkpoint every N steps
  keep_last: 5                    # Keep last N checkpoints
  log_dir: "logs"                 # Directory for TensorBoard logs
  
  # Checkpoint format
  save_format: "torch"            # "torch" or "safetensors"
  save_optimizer: true            # Save optimizer state
  save_scheduler: true            # Save scheduler state

# Generation
generation:
  temperature: 0.8                # Sampling temperature (0.0-2.0)
  top_k: 50                       # Top-k sampling (0 = disabled)
  top_p: 0.95                     # Top-p (nucleus) sampling (0.0-1.0)
  repetition_penalty: 1.0         # Repetition penalty (1.0 = no penalty)
  max_length: 512                 # Maximum generation length
  min_length: 10                  # Minimum generation length
  do_sample: true                 # Use sampling (false = greedy)
  num_beams: 1                    # Beam search width (1 = no beam search)
  early_stopping: true            # Stop generation when EOS is reached

# Logging
logging:
  level: "INFO"                   # Logging level
  format: "detailed"              # "simple" or "detailed"
  log_file: "logs/training.log"   # Log file path
  wandb_project: null             # Weights & Biases project name
  wandb_entity: null              # Weights & Biases entity
```

### Environment Variables

```bash
# Training
export HRMLM_CACHE_DIR="~/.cache/hrmlm"
export HRMLM_LOG_LEVEL="INFO"
export HRMLM_DEVICE="cuda"

# Distributed Training
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0

# Mixed Precision
export AMP_ENABLED="1"
export AMP_OPT_LEVEL="O2"
```

## Training

### Data Preparation

```python
from data.prepare_dataset import prepare_dataset

# Prepare dataset from raw text
prepare_dataset(
    input_file="data/raw/corpus.txt",
    output_dir="data/processed",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)

# Or use command line
python -m data.prepare_dataset \
    --input data/raw/corpus.txt \
    --output data/processed \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### Training Script

```python
import torch
from train import Trainer

# Initialize trainer
trainer = Trainer(config_path="config.yaml")

# Optionally resume from checkpoint
trainer.load_checkpoint("checkpoints/epoch_3.pth")

# Start training
trainer.train()

# Evaluate on test set
test_loss = trainer.evaluate(test=True)
print(f"Test loss: {test_loss:.4f}")
```

### Monitoring Training

```bash
# TensorBoard
tensorboard --logdir logs --port 6006

# View logs
tail -f logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Training Recipes

#### Small Model (125M parameters)
```yaml
model:
  n_ctx: 512
  n_embd: 768
  n_hidden: 1024
  n_high_hidden: 768
  T: 3
  N: 2

training:
  batch_size: 32
  learning_rate: 3e-4
  warmup_steps: 1000
```

#### Medium Model (350M parameters)
```yaml
model:
  n_ctx: 1024
  n_embd: 1024
  n_hidden: 1536
  n_high_hidden: 1024
  T: 5
  N: 3

training:
  batch_size: 16
  learning_rate: 2e-4
  warmup_steps: 2000
  gradient_accumulation_steps: 8
```

#### Large Model (1B parameters)
```yaml
model:
  n_ctx: 2048
  n_embd: 2048
  n_hidden: 2560
  n_high_hidden: 2048
  T: 7
  N: 4

training:
  batch_size: 8
  learning_rate: 1e-4
  warmup_steps: 5000
  gradient_accumulation_steps: 16
  mixed_precision: "bf16"
```

## Inference

### Basic Usage

```python
import torch
from model.hrmlm import HRMLM
from model.tokenizer import Tokenizer

# Load model and tokenizer
model = HRMLM.from_pretrained("checkpoints/best_model")
tokenizer = Tokenizer("models/tokenizer.model")

# Move to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Single generation
prompt = "The future of artificial intelligence"
input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
input_tensor = torch.tensor([input_ids], device=device)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_tensor,
        max_length=100,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

generated_text = tokenizer.decode(generated_ids[0].tolist())
print(generated_text)
```

### Batch Inference

```python
from inference import BatchGenerator

# Initialize batch generator
generator = BatchGenerator(
    checkpoint_path="checkpoints/best_model.pth",
    config_path="config.yaml",
    batch_size=8
)

# Generate from multiple prompts
prompts = [
    "The meaning of life is",
    "In a galaxy far away",
    "The secret to happiness"
]

results = generator.generate_batch(
    prompts=prompts,
    max_length=100,
    temperature=0.8,
    top_p=0.9
)

for i, (prompt, generated) in enumerate(zip(prompts, results)):
    print(f"Prompt {i+1}: {prompt}")
    print(f"Generated: {generated}")
    print("-" * 50)
```

### Web Server

```python
# server.py
from flask import Flask, request, jsonify
import torch
from model.hrmlm import HRMLM
from model.tokenizer import Tokenizer

app = Flask(__name__)

# Load model
model = HRMLM.from_pretrained("checkpoints/best_model")
tokenizer = Tokenizer("models/tokenizer.model")
model.eval()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.8)
    
    # Encode and generate
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_tensor,
            max_length=max_length,
            temperature=temperature
        )
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return jsonify({
        'prompt': prompt,
        'generated': generated_text,
        'tokens': len(generated_ids[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

## API Reference

### HRMLM Class

```python
class HRMLM(nn.Module):
    """
    Hierarchical Recurrent Memory Language Model.
    
    Args:
        vocab_size (int): Size of vocabulary
        n_ctx (int): Maximum context length
        n_embd (int): Embedding dimension
        n_hidden (int): Low-level hidden dimension
        n_high_hidden (int): High-level hidden dimension
        T (int): Low-level cycles per step
        N (int): High-level cycles per token
        dropout (float): Dropout rate (0.0-1.0)
        layer_norm (bool): Whether to use layer normalization
    
    Methods:
        forward(input_ids, past_states=None, return_states=False)
        generate(prompt_ids, max_length=100, temperature=1.0,
                 top_k=50, top_p=1.0, eos_token_id=None)
        from_pretrained(path, device=None, **kwargs)
        save_pretrained(path, save_config=True)
    """
```

### Tokenizer Class

```python
class Tokenizer:
    """
    SentencePiece tokenizer wrapper.
    
    Args:
        model_path (str): Path to SentencePiece model file
    
    Methods:
        encode(text, add_bos=True, add_eos=True, max_length=None)
        decode(token_ids, skip_special_tokens=True)
        train(input_file, model_prefix, vocab_size=32000,
              model_type="bpe", character_coverage=1.0)
        get_vocab()
        token_to_id(token)
        id_to_token(id)
    """
```

### Trainer Class

```python
class Trainer:
    """
    Main training class.
    
    Args:
        config_path (str): Path to configuration YAML file
        resume_from (str, optional): Path to checkpoint to resume from
    
    Methods:
        train()
        evaluate(test=False)
        save_checkpoint(epoch, loss)
        load_checkpoint(path)
        generate_samples(num_samples=5, max_length=100)
    """
```

## Examples

### Example 1: Training on Custom Dataset

```python
import yaml
from train import Trainer

# Custom configuration
config = {
    "model": {
        "vocab_size": 50000,
        "n_ctx": 512,
        "n_embd": 512,
        "n_hidden": 768,
        "n_high_hidden": 512,
        "T": 4,
        "N": 2,
        "dropout": 0.1
    },
    "dataset": {
        "train_file": "data/my_corpus_train.txt",
        "val_file": "data/my_corpus_val.txt",
        "sequence_length": 512,
        "batch_size": 16
    },
    "training": {
        "epochs": 20,
        "learning_rate": 2e-4,
        "warmup_steps": 1000
    }
}

# Save configuration
with open("my_config.yaml", "w") as f:
    yaml.dump(config, f)

# Train
trainer = Trainer(config_path="my_config.yaml")
trainer.train()
```

### Example 2: Fine-tuning on Domain-Specific Text

```python
from model.hrmlm import HRMLM
from model.tokenizer import Tokenizer
import torch.optim as optim

# Load pretrained model
model = HRMLM.from_pretrained("checkpoints/pretrained_model")
tokenizer = Tokenizer("models/tokenizer.model")

# Freeze some layers
for param in model.embedding.parameters():
    param.requires_grad = False

# Domain-specific fine-tuning
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=0.01
)

# Training loop
for epoch in range(10):
    for batch in domain_specific_dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    # Save fine-tuned model
    torch.save(model.state_dict(), f"checkpoints/finetuned_epoch_{epoch}.pth")
```

### Example 3: Using HRMLM as a Feature Extractor

```python
import torch
from model.hrmlm import HRMLM

class TextClassifier(nn.Module):
    def __init__(self, hrmlm_model, num_classes):
        super().__init__()
        self.hrmlm = hrmlm_model
        
        # Freeze HRMLM parameters
        for param in self.hrmlm.parameters():
            param.requires_grad = False
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hrmlm_model.n_high_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids):
        # Get hidden states from HRMLM
        with torch.no_grad():
            _, (h_l, h_h) = self.hrmlm(input_ids, return_states=True)
        
        # Use high-level hidden state for classification
        features = h_h.mean(dim=0)  # Pool across sequence
        
        # Classify
        logits = self.classifier(features)
        return logits

# Usage
model = HRMLM.from_pretrained("checkpoints/pretrained_model")
classifier = TextClassifier(model, num_classes=10)
```

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error during training

**Solutions**:
```python
# Reduce batch size
config['dataset']['batch_size'] = 8

# Use gradient accumulation
config['optimization']['gradient_accumulation_steps'] = 8

# Use mixed precision
config['optimization']['mixed_precision'] = "fp16"

# Use gradient checkpointing (if implemented)
model.use_gradient_checkpointing = True

# Clear cache
torch.cuda.empty_cache()
```

#### Issue 2: Slow Training

**Symptoms**: Low GPU utilization, slow iteration times

**Solutions**:
```python
# Increase DataLoader workers
config['dataset']['num_workers'] = 8

# Enable pinned memory
config['dataset']['pin_memory'] = True

# Use torch.compile (PyTorch 2.0+)
config['optimization']['compile_model'] = True

# Use larger batch size with gradient accumulation
config['dataset']['batch_size'] = 32
config['optimization']['gradient_accumulation_steps'] = 4
```

#### Issue 3: NaN Loss

**Symptoms**: Loss becomes NaN during training

**Solutions**:
```python
# Reduce learning rate
config['training']['learning_rate'] = 1e-4

# Add gradient clipping
config['training']['grad_clip'] = 1.0

# Use gradient norm scaling for mixed precision
scaler = torch.cuda.amp.GradScaler()

# Check for NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
```

#### Issue 4: Poor Generation Quality

**Symptoms**: Repetitive or nonsensical text generation

**Solutions**:
```python
# Adjust sampling parameters
generation_params = {
    'temperature': 0.7,  # Lower for more focused, higher for more creative
    'top_k': 40,         # Limit to top-k tokens
    'top_p': 0.9,        # Use nucleus sampling
    'repetition_penalty': 1.2  # Penalize repetition
}

# Use beam search
generation_params['num_beams'] = 5
generation_params['early_stopping'] = True

# Ensure model is in eval mode
model.eval()
```

### Debugging Commands

```bash
# Check GPU memory
nvidia-smi

# Monitor training progress
watch -n 1 "tail -n 20 logs/training.log"

# Check for NaN values
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Profile model
python -m cProfile -o profile.stats train.py --config config.yaml
```

## Citation

If you use HRMLM in your research, please cite:

```bibtex
@software{hrmlm2023,
  title = {HRMLM: Hierarchical Recurrent Memory Language Model},
  year = {2025},
  url = {https://github.com/aritrodium/hrmlm},
  version = {1.0.0}
}
```

## License

```
MIT License

Copyright (c) 2023 Aritra Roy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8 .
```

## Support

- 📖 **Documentation**: [https://github.com/yourusername/hrmlm/wiki](https://github.com/aritrodium/hrmlm/wiki)
- 🐛 **Issues**: [GitHub Issues](https://github.com/aritrodium/hrmlm/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/aritrodium/hrmlm/discussions)


---
