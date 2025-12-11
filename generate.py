"""
Text generation script.
"""

import torch
import yaml
from model.hrmlm import HRMLM
from model.tokenizer import Tokenizer
import argparse


def load_model(checkpoint_path: str, config_path: str = None):
    """Load model from checkpoint."""
    if config_path is None:
        config_path = "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    tokenizer = Tokenizer("models/tokenizer.model")
    
    # Initialize model
    model = HRMLM(
        vocab_size=tokenizer.vocab_size,
        n_ctx=config['model']['n_ctx'],
        n_embd=config['model']['n_embd'],
        n_hidden=config['model']['n_hidden'],
        n_high_hidden=config['model']['n_high_hidden'],
        T=config['model']['T'],
        N=config['model']['N'],
        dropout=0.0,  # Disable dropout for generation
        layer_norm=config['model']['layer_norm']
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, **kwargs):
    """Generate text from prompt."""
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=kwargs.get('max_length', 100),
            temperature=kwargs.get('temperature', 0.8),
            top_k=kwargs.get('top_k', 50),
            top_p=kwargs.get('top_p', 0.95),
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with HRMLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=200,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p (nucleus) sampling parameter")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.checkpoint)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate text
    generated = generate_text(
        model, 
        tokenizer, 
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print("\n" + "="*50)
    print("Generated Text:")
    print("="*50)
    print(generated)
