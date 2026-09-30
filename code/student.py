"""小型 GPT；可按配置单独比较前馈层和归一化层。"""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """按特征的均方根缩放，不减去均值。"""

    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(x.dtype) * self.weight


def make_norm(width, kind):
    if kind == 'layernorm':
        return nn.LayerNorm(width)
    if kind == 'rmsnorm':
        return RMSNorm(width)
    raise ValueError(f'Unknown normalization type: {kind}')


class RotaryEmbedding(nn.Module):
    """对每个注意力头的 Q、K 施加无训练参数的位置旋转。"""

    def __init__(self, head_dim, context, fraction=1.0, base=10000):
        super().__init__()
        if not 0.0 < fraction <= 1.0:
            raise ValueError('RoPE fraction must be in (0, 1].')
        rotary_dim = int(head_dim * fraction)
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError('The rotary dimension must be a positive even number.')
        self.rotary_dim = rotary_dim
        inv_freq = base ** (-torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        angles = torch.outer(torch.arange(context).float(), inv_freq)
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    @staticmethod
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def forward(self, q, k):
        length = q.shape[-2]
        cos, sin = self.cos[..., :length, :], self.sin[..., :length, :]
        q_rot, k_rot = q[..., :self.rotary_dim], k[..., :self.rotary_dim]
        q_float, k_float = q_rot.float(), k_rot.float()
        q_rot = (q_float * cos + self.rotate_half(q_float) * sin).to(q.dtype)
        k_rot = (k_float * cos + self.rotate_half(k_float) * sin).to(k.dtype)
        q = torch.cat((q_rot, q[..., self.rotary_dim:]), dim=-1)
        k = torch.cat((k_rot, k[..., self.rotary_dim:]), dim=-1)
        return q, k


class SwiGLU(nn.Module):
    """带门控的前馈层，隐藏宽度约为 8/3 倍以接近 GELU 的参数量。"""

    def __init__(self, width):
        super().__init__()
        hidden = 8 * round(width / 3)
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.proj = nn.Linear(hidden, width)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.value(x))


class Block(nn.Module):
    """Pre-norm、因果自注意力和前馈网络组成的 Transformer block。"""

    def __init__(self, width, heads, kv_heads, ffn, norm, position, context, dropout,
                 attention_dropout, rope_fraction, window_size, gated_attention):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.window_size = window_size
        self.norm1 = make_norm(width, norm)
        self.norm2 = make_norm(width, norm)
        if kv_heads == heads:
            self.qkv = nn.Linear(width, 3 * width)
        else:
            self.q_proj = nn.Linear(width, width)
            self.k_proj = nn.Linear(width, kv_heads * (width // heads))
            self.v_proj = nn.Linear(width, kv_heads * (width // heads))
        self.proj = nn.Linear(width, width)
        head_dim = width // heads
        self.q_norm = nn.LayerNorm(head_dim) if gated_attention else None
        self.k_norm = nn.LayerNorm(head_dim) if gated_attention else None
        self.attn_gate = nn.Linear(width, width) if gated_attention else None
        self.rope = (RotaryEmbedding(width // heads, context, rope_fraction)
                     if position == 'rope' else None)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout

        if ffn == 'swiglu':
            self.mlp = SwiGLU(width)
        elif ffn == 'gelu':
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width),
                nn.GELU(),
                nn.Linear(4 * width, width),
            )
        else:
            raise ValueError(f'Unknown feed-forward type: {ffn}')

    def forward(self, x):
        batch, length, width = x.shape
        normalized = self.norm1(x)
        if self.kv_heads == self.heads:
            qkv = self.qkv(normalized).view(batch, length, 3, self.heads, width // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
        else:
            head_dim = width // self.heads
            q = self.q_proj(normalized).view(batch, length, self.heads, head_dim).transpose(1, 2)
            k = self.k_proj(normalized).view(batch, length, self.kv_heads, head_dim).transpose(1, 2)
            v = self.v_proj(normalized).view(batch, length, self.kv_heads, head_dim).transpose(1, 2)
            groups = self.heads // self.kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        attn_mask = None
        if self.window_size is not None and length > self.window_size:
            positions = torch.arange(length, device=x.device)
            distance = positions[:, None] - positions[None, :]
            attn_mask = (distance >= 0) & (distance < self.window_size)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None,
            dropout_p=self.attention_dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        attn_output = self.proj(attended)
        if self.attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(normalized))
        x = x + self.dropout(attn_output)
        return x + self.dropout(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    """返回训练用 logits 和评估用归一化 log probabilities。"""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        vocab = config['vocab']
        ffn = config.get('ffn', 'swiglu')
        norm = config.get('norm', 'layernorm')
        position = config.get('position', 'learned')
        dropout = float(config.get('dropout', 0.0))
        attention_dropout = float(config.get('attention_dropout', 0.0))
        rope_fraction = float(config.get('rope_fraction', 1.0))
        kv_heads = int(config.get('kv_heads', config['heads']))
        window_size = config.get('window_size')
        gated_attention = bool(config.get('gated_attention', False))
        if position not in ('learned', 'rope'):
            raise ValueError(f'Unknown position type: {position}')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('Dropout must be in [0, 1).')
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError('Attention dropout must be in [0, 1).')
        if config['width'] % config['heads'] or config['heads'] % kv_heads:
            raise ValueError('Width must divide evenly across heads and query heads across KV heads.')
        if window_size is not None and int(window_size) < 1:
            raise ValueError('Attention window must be positive.')

        self.token = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(self.context, width) if position == 'learned' else None
        self.drop_emb = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(width, config['heads'], kv_heads, ffn, norm, position, self.context, dropout,
                   attention_dropout, rope_fraction, window_size, gated_attention)
             for _ in range(config['depth'])]
        )
        self.norm = make_norm(width, norm)
        self.head = nn.Linear(width, vocab, bias=False)

        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        length = ids.shape[1]
        x = self.token(ids)
        if self.pos is not None:
            positions = torch.arange(length, device=ids.device)
            x = x + self.pos(positions)
        x = self.drop_emb(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    if config.get('normalization_variant', 'control') != 'control' or float(config.get('temperature', 1.)) != 1.:
        from experimental_models import build_model as build_experimental_model
        return build_experimental_model(config)
    return GPT(config)
