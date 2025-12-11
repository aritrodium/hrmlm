"""
Hierarchical Recurrent Memory Language Model (HRMLM)
Implementation of a hierarchical RNN with multi-timescale processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNormGRUCell(nn.Module):
    """GRU Cell with Layer Normalization."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates: reset, update, new
        self.W_ir = nn.Linear(input_dim, hidden_dim)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_iz = nn.Linear(input_dim, hidden_dim)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.ln_ir = nn.LayerNorm(hidden_dim)
        self.ln_hr = nn.LayerNorm(hidden_dim)
        self.ln_iz = nn.LayerNorm(hidden_dim)
        self.ln_hz = nn.LayerNorm(hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.ln_hn = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Reset gate
        r = torch.sigmoid(
            self.ln_ir(self.W_ir(x)) + 
            self.ln_hr(self.W_hr(h))
        )
        
        # Update gate
        z = torch.sigmoid(
            self.ln_iz(self.W_iz(x)) + 
            self.ln_hz(self.W_hz(h))
        )
        
        # New gate
        n = torch.tanh(
            self.ln_in(self.W_in(x)) + 
            r * self.ln_hn(self.W_hn(h))
        )
        
        # New hidden state
        h_new = (1 - z) * n + z * h
        return self.dropout(h_new)


class LowLevelRNN(nn.Module):
    """Low-level RNN operating at fast timescale."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cell = LayerNormGRUCell(input_dim + hidden_dim, hidden_dim, dropout)
        
    def forward(self, x: torch.Tensor, h_l: torch.Tensor, h_h: torch.Tensor) -> torch.Tensor:
        # Concatenate input with high-level hidden state
        combined = torch.cat([x, h_h], dim=-1)
        return self.cell(combined, h_l)


class HighLevelRNN(nn.Module):
    """High-level RNN operating at slow timescale."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cell = LayerNormGRUCell(input_dim, hidden_dim, dropout)
        
    def forward(self, h_h_prev: torch.Tensor, h_l: torch.Tensor) -> torch.Tensor:
        return self.cell(h_l, h_h_prev)


class HRMLM(nn.Module):
    """
    Hierarchical Recurrent Memory Language Model.
    
    Args:
        vocab_size: Size of vocabulary
        n_ctx: Context window size (maximum sequence length)
        n_embd: Embedding dimension
        n_hidden: Low-level hidden dimension
        n_high_hidden: High-level hidden dimension
        T: Number of low-level cycles per step
        N: Number of high-level cycles per token
        dropout: Dropout rate
        layer_norm: Whether to use layer normalization
    """
    
    def __init__(self,
                 vocab_size: int,
                 n_ctx: int = 1024,
                 n_embd: int = 768,
                 n_hidden: int = 1024,
                 n_high_hidden: int = 768,
                 T: int = 5,
                 N: int = 3,
                 dropout: float = 0.1,
                 layer_norm: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.n_high_hidden = n_high_hidden
        self.T = T
        self.N = N
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Hierarchical RNN layers
        self.low_rnn = LowLevelRNN(n_embd, n_hidden, dropout)
        self.high_rnn = HighLevelRNN(n_hidden, n_high_hidden, dropout)
        
        # Output projection
        self.output_proj = nn.Linear(n_high_hidden, vocab_size)
        
        # Layer normalization for embeddings
        self.ln_emb = nn.LayerNorm(n_embd) if layer_norm else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self,
                input_ids: torch.Tensor,
                past_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_states: bool = False):
        """
        Forward pass through HRMLM.
        
        Args:
            input_ids: Input token indices [batch_size, seq_len]
            past_states: Tuple of (h_l, h_h) from previous forward pass
            return_states: Whether to return hidden states for next step
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            states: Optional tuple of final hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize hidden states
        if past_states is None:
            device = input_ids.device
            h_l = torch.zeros(batch_size, self.n_hidden, device=device)
            h_h = torch.zeros(batch_size, self.n_high_hidden, device=device)
        else:
            h_l, h_h = past_states
        
        # Store all logits
        logits_list = []
        
        # Process sequence
        for t in range(seq_len):
            # Get current token embedding
            x = self.embedding(input_ids[:, t])
            x = self.ln_emb(x)
            x = self.dropout(x)
            
            # Hierarchical processing
            for _ in range(self.N):
                # Low-level processing (fast timescale)
                for _ in range(self.T):
                    h_l = self.low_rnn(x, h_l, h_h)
                
                # High-level processing (slow timescale)
                h_h = self.high_rnn(h_h, h_l)
            
            # Generate logits for current position
            logits_t = self.output_proj(h_h)
            logits_list.append(logits_t.unsqueeze(1))
        
        # Combine logits
        logits = torch.cat(logits_list, dim=1)
        
        if return_states:
            return logits, (h_l, h_h)
        return logits
    
    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 eos_token_id: Optional[int] = None):
        """
        Generate text from prompt.
        
        Args:
            prompt_ids: Starting tokens [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            eos_token_id: Token ID for stopping generation
            
        Returns:
            generated_ids: Generated token IDs
        """
        self.eval()
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        
        # Initialize with prompt
        generated_ids = prompt_ids.clone()
        current_input = prompt_ids
        
        # Initialize hidden states
        h_l = torch.zeros(batch_size, self.n_hidden, device=device)
        h_h = torch.zeros(batch_size, self.n_high_hidden, device=device)
        
        for _ in range(max_length):
            # Get logits for last token
            logits, (h_l, h_h) = self(current_input, (h_l, h_h), return_states=True)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply nucleus sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_input = next_token
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
        
        return generated_ids
