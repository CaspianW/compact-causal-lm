"""QKNorm and a SwiGLU adaptation of NormFormer, based on the local student GPT."""
import math
import torch
from torch import nn
from torch.nn import functional as F
import student


class NormalizedBlock(nn.Module):
    def __init__(self, original, variant, width, context):
        super().__init__()
        self.variant = variant
        self.heads = original.heads
        self.norm1, self.norm2 = original.norm1, original.norm2
        self.qkv, self.proj, self.mlp = original.qkv, original.proj, original.mlp
        self.rope, self.dropout = original.rope, original.dropout
        self.attention_dropout = original.attention_dropout
        if variant == 'qknorm':
            self.qk_scale = nn.Parameter(torch.tensor(math.log2(context * context - context)))
        else:
            self.head_scale = nn.Parameter(torch.ones(self.heads))
            self.attention_norm = nn.LayerNorm(width)
            self.hidden_norm = nn.LayerNorm(original.mlp.gate.out_features)

    def forward(self, x):
        batch, length, width = x.shape
        normalized = self.norm1(x)
        qkv = self.qkv(normalized).view(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q, k)
        if self.variant == 'qknorm':
            q = F.normalize(q.float(), dim=-1, eps=1e-6).to(q.dtype)
            k = F.normalize(k.float(), dim=-1, eps=1e-6).to(k.dtype)
            q = q * self.qk_scale.to(q.dtype)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.,
            scale=1.0 if self.variant == 'qknorm' else None)
        if self.variant == 'normformer':
            attended = attended * self.head_scale.to(attended.dtype)[None, :, None, None]
        output = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        if self.variant == 'normformer':
            output = self.attention_norm(output)
        x = x + self.dropout(output)
        normalized = self.norm2(x)
        if self.variant == 'normformer':
            hidden = F.silu(self.mlp.gate(normalized)) * self.mlp.value(normalized)
            output = self.mlp.proj(self.hidden_norm(hidden))
        else:
            output = self.mlp(normalized)
        return x + self.dropout(output)


class NormalizedGPT(student.GPT):
    def __init__(self, config):
        super().__init__(config)
        variant = config.get('normalization_variant', 'control')
        if variant not in ('control', 'qknorm', 'normformer'):
            raise ValueError('Unknown normalization variant.')
        if (config.get('ffn') != 'swiglu' or config.get('kv_heads', config['heads']) != config['heads']
                or config.get('window_size') is not None or config.get('gated_attention', False)):
            raise ValueError('This experiment requires dense MHA and SwiGLU without attention gating.')
        if variant != 'control':
            # Reuse the initialized modules so common parameters match across all three variants.
            self.blocks = nn.ModuleList([
                NormalizedBlock(block, variant, config['width'], config['context'])
                for block in self.blocks])
        self.temperature = float(config.get('temperature', 1.))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Temperature must be finite and positive.')

    def forward(self, ids):
        logits = super().forward(ids)
        return logits if self.temperature == 1. else logits.float() / self.temperature


def build_model(config):
    return NormalizedGPT(config)
