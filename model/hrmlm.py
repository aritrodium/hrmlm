"""
Hierarchical Recurrent Memory Language Model with SFT/DPO support.
Complete implementation for instruction following and preference optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import warnings


class LayerNormGRUCell(nn.Module):
    """GRU Cell with Layer Normalization for stable training."""
    
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRU cell."""
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
        """Process input with high-level context."""
        combined = torch.cat([x, h_h], dim=-1)
        return self.cell(combined, h_l)


class HighLevelRNN(nn.Module):
    """High-level RNN operating at slow timescale."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cell = LayerNormGRUCell(input_dim, hidden_dim, dropout)
        
    def forward(self, h_h_prev: torch.Tensor, h_l: torch.Tensor) -> torch.Tensor:
        """Integrate low-level information."""
        return self.cell(h_l, h_h_prev)


class HRMLM(nn.Module):
    """
    Hierarchical Recurrent Memory Language Model.
    
    Supports:
    - Pretraining (causal language modeling)
    - Supervised Fine-Tuning (SFT)
    - Direct Preference Optimization (DPO)
    - Reward modeling
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
        
        # Layer normalization
        self.ln_emb = nn.LayerNorm(n_embd) if layer_norm else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using appropriate methods."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False,
                return_dict: bool = True):
        """
        Forward pass through HRMLM.
        
        Args:
            input_ids: Input token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling [batch_size, seq_len]
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dictionary output
            
        Returns:
            Dict or tuple containing:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - loss: Optional language modeling loss
            - hidden_states: Optional hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize hidden states
        device = input_ids.device
        h_l = torch.zeros(batch_size, self.n_hidden, device=device)
        h_h = torch.zeros(batch_size, self.n_high_hidden, device=device)
        
        # Store all logits and hidden states
        logits_list = []
        hidden_states_list = [] if output_hidden_states else None
        
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
            
            # Store hidden states if requested
            if output_hidden_states:
                hidden_states_list.append(h_h.unsqueeze(1))
        
        # Combine outputs
        logits = torch.cat(logits_list, dim=1)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Prepare output
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            if output_hidden_states:
                hidden_states = torch.cat(hidden_states_list, dim=1) if hidden_states_list else None
                output = output + (hidden_states,)
            return output
        
        output_dict = {
            "logits": logits,
            "loss": loss,
        }
        
        if output_hidden_states:
            hidden_states = torch.cat(hidden_states_list, dim=1) if hidden_states_list else None
            output_dict["hidden_states"] = hidden_states
        
        return output_dict
    
    def get_log_probs(self, 
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      past_key_values: Optional[Tuple] = None,
                      position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get log probabilities for DPO training.
        
        Args:
            input_ids: Input token indices
            attention_mask: Attention mask
            past_key_values: Past hidden states for generation
            position_ids: Position indices
            
        Returns:
            Log probabilities [batch_size]
        """
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probabilities of input tokens
        log_probs_selected = torch.gather(
            log_probs[:, :-1], 
            dim=2, 
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            log_probs_selected = log_probs_selected * attention_mask[:, :-1]
        
        # Sum log probabilities along sequence dimension
        sum_log_probs = log_probs_selected.sum(dim=1)
        
        return sum_log_probs
    
    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 do_sample: bool = True,
                 num_beams: int = 1,
                 attention_mask: Optional[torch.Tensor] = None,
                 **kwargs):
        """
        Generate text from prompt.
        
        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            attention_mask: Attention mask
            
        Returns:
            generated_ids: Generated token IDs [batch_size, generated_seq_len]
        """
        self.eval()
        
        if num_beams > 1:
            return self._beam_search_generate(
                input_ids, max_length, num_beams,
                attention_mask, **kwargs
            )
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize with input
        generated = input_ids
        
        # Initialize hidden states
        h_l = torch.zeros(batch_size, self.n_hidden, device=device)
        h_h = torch.zeros(batch_size, self.n_high_hidden, device=device)
        
        for _ in range(max_length):
            # Get logits for last token
            with torch.no_grad():
                x = self.embedding(generated[:, -1])
                x = self.ln_emb(x)
                
                # Hierarchical processing for last token
                for _ in range(self.N):
                    for _ in range(self.T):
                        h_l = self.low_rnn(x, h_l, h_h)
                    h_h = self.high_rnn(h_h, h_l)
                
                logits = self.output_proj(h_h)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Apply nucleus sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def _apply_repetition_penalty(self, logits, generated, penalty):
        """Apply repetition penalty to logits."""
        for i in range(generated.shape[0]):
            for token_id in set(generated[i].tolist()):
                logits[i, token_id] = logits[i, token_id] / penalty
        return logits
    
    def _beam_search_generate(self, input_ids, max_length, num_beams, attention_mask, **kwargs):
        """Beam search generation."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize beams
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9
        
        generated = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        
        for step in range(max_length):
            # Flatten for processing
            flat_generated = generated.view(batch_size * num_beams, -1)
            
            # Get logits
            with torch.no_grad():
                # This is simplified - in practice you'd need to manage hidden states per beam
                outputs = self(flat_generated)
                next_token_logits = outputs.logits[:, -1, :]
            
            # Calculate scores
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores.view(-1, 1)
            
            # Reshape for beam selection
            next_token_scores = next_token_scores.view(batch_size, num_beams * self.vocab_size)
            
            # Select top beams
            topk_scores, topk_indices = torch.topk(next_token_scores, num_beams, dim=-1)
            
            # Update beams
            beam_indices = topk_indices // self.vocab_size
            token_indices = topk_indices % self.vocab_size
            
            # Update generated sequences
            new_generated = []
            for i in range(batch_size):
                batch_generated = []
                for j in range(num_beams):
                    beam_idx = beam_indices[i, j]
                    token_idx = token_indices[i, j]
                    batch_generated.append(
                        torch.cat([generated[i, beam_idx], token_idx.unsqueeze(0)])
                    )
                new_generated.append(torch.stack(batch_generated))
            
            generated = torch.stack(new_generated)
            beam_scores = topk_scores
        
        # Return best beam for each batch
        best_beams = torch.argmax(beam_scores, dim=-1)
        best_generated = []
        for i in range(batch_size):
            best_generated.append(generated[i, best_beams[i]])
        
        return torch.stack(best_generated)
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "n_ctx": self.n_ctx,
            "n_embd": self.n_embd,
            "n_hidden": self.n_hidden,
            "n_high_hidden": self.n_high_hidden,
            "T": self.T,
            "N": self.N,
            "dropout": self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0,
            "layer_norm": isinstance(self.ln_emb, nn.LayerNorm)
        }
        
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        """Load model from pretrained weights."""
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(pretrained_model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            vocab_size=config["vocab_size"],
            n_ctx=config.get("n_ctx", 1024),
            n_embd=config.get("n_embd", 768),
            n_hidden=config.get("n_hidden", 1024),
            n_high_hidden=config.get("n_high_hidden", 768),
            T=config.get("T", 5),
            N=config.get("N", 3),
            dropout=config.get("dropout", 0.1),
            layer_norm=config.get("layer_norm", True)
        )
        
        # Load weights
        weights_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        return model


class RewardModel(nn.Module):
    """
    Reward model for DPO training.
    Takes the HRMLM and adds a reward head.
    """
    
    def __init__(self, base_model: HRMLM):
        super().__init__()
        self.base_model = base_model
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.n_high_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass to compute reward scores."""
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use last hidden state for reward prediction
        last_hidden_state = outputs.hidden_states[:, -1, :] if outputs.hidden_states is not None else outputs.logits[:, -1, :]
        
        # Compute reward
        reward = self.reward_head(last_hidden_state)
        
        return reward
