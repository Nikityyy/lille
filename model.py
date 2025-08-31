import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Union, List

@dataclass
class GPTConfig:
    """
    Configuration for the GPT model, inspired by GPT-OSS/Llama but adapted for this project.
    """
    n_embd: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 32000
    block_size: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    n_kv_heads: Optional[int] = 4
    rope_theta: float = 10000.0

    def to_dict(self):
        return asdict(self)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization, as used in GPT-OSS and Llama.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Efficiently repeat the key and value tensors for Grouped-Query Attention.
    [B, n_kv_heads, T, head_dim] -> [B, n_q_heads, T, head_dim]
    """
    B, n_kv_heads, T, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, head_dim)
        .reshape(B, n_kv_heads * n_rep, T, head_dim)
    )

class RotaryPositionalEmbedding(nn.Module):
    """
    Original RoPE implementation, kept for its efficiency in training.
    """
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len: int):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    """
    Attention module with pre-normalization, based on Llama/GPT-OSS style.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_q_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = config.n_embd // self.n_q_heads

        self.qkv_proj = nn.Linear(config.n_embd, (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        
        q_heads_concat_dim = self.n_q_heads * self.head_dim
        self.out_proj = nn.Linear(q_heads_concat_dim, config.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.n_embd, eps=config.layer_norm_eps)
        self.out_proj.GPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor], past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        h = self.norm(x)

        qkv = self.qkv_proj(h)
        q_len = self.n_q_heads * self.head_dim
        k_len = self.n_kv_heads * self.head_dim

        q = qkv[..., :q_len].view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_len : q_len + k_len].view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_len + k_len :].view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = rotary_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_kv = (k.to(x.dtype), v.to(x.dtype))

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        is_causal_for_sdpa = False

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal_for_sdpa,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.out_proj(y)

        return x + y, present_kv

class FeedForward(nn.Module):
    """
    FeedForward block with pre-normalization and SwiGLU, based on Llama/GPT-OSS style.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        multiple_of = 256
        hidden_dim = multiple_of * round(hidden_dim / multiple_of)

        self.norm = RMSNorm(config.n_embd, eps=config.layer_norm_eps)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

        self.down_proj.GPT_SCALE_INIT = 1

    def forward(self, x):
        h = self.norm(x)
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        fused = gate * up
        return x + self.down_proj(fused)

class Block(nn.Module):
    """
    Transformer Block in the Llama/GPT-OSS pre-normalization style.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor], past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, present_kv = self.attention(x, rotary_emb, past_kv, attn_mask=attn_mask)
        out = self.feed_forward(h)
        return out, present_kv

class GPT(nn.Module):
    """
    The main GPT model, composed of the new Llama/GPT-OSS-style blocks.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.rotary_emb = RotaryPositionalEmbedding(config.n_embd // config.n_heads, config.block_size, base=config.rope_theta)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.n_embd, eps=config.layer_norm_eps)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        """
        Returns the model's input embeddings.
        Required by the Hugging Face PreTrainedModel interface.
        """
        return self.tok_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the model's input embeddings.
        Required by the Hugging Face PreTrainedModel interface.
        """
        self.tok_embeddings = new_embeddings

    def forward(self, input_ids: torch.Tensor, past_kv_cache: Optional[list] = None, use_cache: bool = False, attn_mask: Optional[torch.Tensor] = None) -> tuple:
        B, T = input_ids.size()
        seq_len_offset = past_kv_cache[0][0].shape[2] if past_kv_cache is not None else 0
        total_sequence_length = seq_len_offset + T

        q_indices = torch.arange(T, device=input_ids.device) + seq_len_offset
        k_indices = torch.arange(total_sequence_length, device=input_ids.device)
        causal_mask = q_indices.unsqueeze(1) >= k_indices.unsqueeze(0)

        if attn_mask is not None:
            padding_mask = attn_mask[:, :total_sequence_length]
            combined_mask = causal_mask.unsqueeze(0) & padding_mask.unsqueeze(1)
        else:
            combined_mask = causal_mask.unsqueeze(0)
        
        final_sdpa_mask = combined_mask.unsqueeze(1)

        h = self.tok_embeddings(input_ids)
        
        cos, sin = self.rotary_emb(h, seq_len=total_sequence_length)
        cos = cos[:, :, seq_len_offset:, :]
        sin = sin[:, :, seq_len_offset:, :]
        rotary_emb = (cos, sin)

        present_kv_cache = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_cache[i] if past_kv_cache is not None else None
            h, present_kv = layer(h, rotary_emb, past_kv, attn_mask=final_sdpa_mask)
            present_kv_cache.append(present_kv)

        h = self.norm(h)
        logits = self.lm_head(h)
            
        return logits, present_kv_cache

    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, stop_on_token: Optional[Union[int, List[int]]] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        past_kv_cache = None
        current_attn_mask = attn_mask

        for _ in range(max_new_tokens):
            B, T = idx.shape
            
            if T >= self.config.block_size:
                break

            current_input = idx[:, -1:] if past_kv_cache is not None else idx

            logits, past_kv_cache = self(current_input, past_kv_cache=past_kv_cache, use_cache=True, attn_mask=current_attn_mask)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(F.softmax(logits, dim=-1), descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if current_attn_mask is not None:
                new_mask_col = torch.ones((B, 1), dtype=current_attn_mask.dtype, device=current_attn_mask.device)
                current_attn_mask = torch.cat((current_attn_mask, new_mask_col), dim=1)

            if stop_on_token is not None:
                stop_tokens = stop_on_token if isinstance(stop_on_token, (list, tuple, set)) else [stop_on_token]
                if idx_next.item() in stop_tokens:
                    break
        return idx
