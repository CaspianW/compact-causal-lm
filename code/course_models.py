"""Small, controlled implementations of attention and routing mechanisms from Lectures 2–4."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from student import RotaryEmbedding, SwiGLU


class ExpertMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.value(x))


class SparseMoE(nn.Module):
    """Four routed experts, top two per token, and one shared expert."""
    def __init__(self, width):
        super().__init__()
        self.router = nn.Linear(width, 4)
        self.experts = nn.ModuleList(ExpertMLP(width) for _ in range(4))
        self.shared = ExpertMLP(width)
        self.aux_loss = None

    def forward(self, x):
        batch, length, width = x.shape
        probabilities = self.router(x).float().softmax(dim=-1)
        selected = probabilities.topk(2, dim=-1)
        choices = selected.indices.reshape(-1, 2)
        weights = (selected.values / selected.values.sum(dim=-1, keepdim=True)).reshape(-1, 2)
        flat = x.reshape(-1, width)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_slot = (choices == expert_id).nonzero(as_tuple=False)
            if token_slot.numel():
                token, slot = token_slot.unbind(dim=-1)
                output.index_add_(0, token, expert(flat[token]) * weights[token, slot, None].to(x.dtype))
        frequency = F.one_hot(choices, 4).float().mean(dim=(0, 1)).detach()
        mean_probability = probabilities.mean(dim=(0, 1))
        self.aux_loss = 0.01 * 4 * (frequency * mean_probability).sum()
        return output.view(batch, length, width) + self.shared(x)


class CourseAttention(nn.Module):
    def __init__(self, width, heads, context, method, dropout, sparse_k):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.method = method
        self.dropout = dropout
        self.sparse_k = sparse_k
        self.training_step = 0
        self.aux_loss = None
        if method == 'mla':
            self.rotary_dim = max(2, self.head_dim // 4)
            self.content_dim = self.head_dim - self.rotary_dim
            latent = width // 4
            self.q_content = nn.Linear(width, heads * self.content_dim)
            self.q_rotary = nn.Linear(width, heads * self.rotary_dim)
            self.kv_down = nn.Linear(width, latent)
            self.k_content = nn.Linear(latent, heads * self.content_dim)
            self.v_up = nn.Linear(latent, width)
            self.k_rotary = nn.Linear(width, self.rotary_dim)
            self.rope = RotaryEmbedding(self.rotary_dim, context)
        else:
            self.qkv = nn.Linear(width, 3 * width)
            self.rope = RotaryEmbedding(self.head_dim, context)
        if method == 'dsa':
            self.index_q = nn.Linear(width, 16, bias=False)
            self.index_k = nn.Linear(width, 16, bias=False)
        if method == 'delta':
            self.alpha = nn.Linear(width, heads)
            self.beta = nn.Linear(width, heads)
        self.proj = nn.Linear(width, width)

    def _linear(self, q, k, v):
        q, k = F.elu(q.float()) + 1., F.elu(k.float()) + 1.
        product_prefix = (k.unsqueeze(-1) * v.float().unsqueeze(-2)).cumsum(dim=2)
        key_prefix = k.cumsum(dim=2)
        numerator = torch.einsum('bhtd,bhtde->bhte', q, product_prefix)
        denominator = (q * key_prefix).sum(dim=-1).clamp_min(1e-6)
        return (numerator / denominator.unsqueeze(-1)).to(v.dtype)

    def _delta(self, x, q, k, v):
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        values = v.float()
        alpha = torch.sigmoid(self.alpha(x.float()).transpose(1, 2))
        beta = torch.sigmoid(self.beta(x.float()).transpose(1, 2))
        batch, heads, length, dim = q.shape
        state = torch.zeros(batch, heads, dim, dim, device=x.device)
        outputs = []
        for index in range(length):
            query, key, value = q[:, :, index], k[:, :, index], values[:, :, index]
            predicted = torch.einsum('bhde,bhd->bhe', state, key)
            correction = key.unsqueeze(-1) * (value - predicted).unsqueeze(-2)
            state = alpha[:, :, index, None, None] * state + beta[:, :, index, None, None] * correction
            outputs.append(torch.einsum('bhde,bhd->bhe', state, query))
        return torch.stack(outputs, dim=2).to(v.dtype)

    def _dsa_mask_and_loss(self, x, q, k):
        length = x.size(1)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        index_scores = self.index_q(x.float()) @ self.index_k(x.float()).transpose(-2, -1)
        index_scores = index_scores / 4.
        index_scores = index_scores.masked_fill(~causal, -1e4)
        with torch.no_grad():
            teacher_scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
            teacher = teacher_scores.masked_fill(~causal, -1e4).softmax(dim=-1).mean(dim=1)
        log_index = index_scores.log_softmax(dim=-1)
        self.aux_loss = 0.01 * (teacher * (teacher.clamp_min(1e-12).log() - log_index)).sum(dim=-1).mean()
        if self.training and self.training_step <= 600:
            return None
        chosen = index_scores.topk(min(self.sparse_k, length), dim=-1).indices
        sparse = torch.zeros_like(index_scores, dtype=torch.bool)
        sparse.scatter_(-1, chosen, True)
        return (sparse & causal).unsqueeze(1)

    def forward(self, x):
        batch, length, width = x.shape
        self.aux_loss = None
        if self.method == 'mla':
            q_content = self.q_content(x).view(batch, length, self.heads, self.content_dim).transpose(1, 2)
            q_rotary = self.q_rotary(x).view(batch, length, self.heads, self.rotary_dim).transpose(1, 2)
            latent = self.kv_down(x)
            k_content = self.k_content(latent).view(batch, length, self.heads, self.content_dim).transpose(1, 2)
            k_rotary = self.k_rotary(x).view(batch, length, 1, self.rotary_dim).transpose(1, 2)
            q_rotary, k_rotary = self.rope(q_rotary, k_rotary)
            q = torch.cat((q_content, q_rotary), dim=-1)
            k = torch.cat((k_content, k_rotary.expand(-1, self.heads, -1, -1)), dim=-1)
            v = self.v_up(latent).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        else:
            qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q, k = self.rope(q, k)
        if self.method == 'linear':
            attended = self._linear(q, k, v)
        elif self.method == 'delta':
            attended = self._delta(x, q, k, v)
        else:
            mask = self._dsa_mask_and_loss(x, q, k) if self.method == 'dsa' else None
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=mask is None,
                dropout_p=self.dropout if self.training else 0.)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class CourseBlock(nn.Module):
    def __init__(self, config, attention_method):
        super().__init__()
        width = config['width']
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attention = CourseAttention(width, config['heads'], config['context'], attention_method,
                                         config.get('attention_dropout', 0.), config.get('sparse_k', 128))
        self.mlp = SparseMoE(width) if config.get('method') == 'moe' else SwiGLU(width)
        self.dropout = nn.Dropout(config.get('dropout', 0.))
        self.aux_loss = None

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        losses = [item for item in (self.attention.aux_loss, getattr(self.mlp, 'aux_loss', None))
                  if item is not None]
        self.aux_loss = sum(losses) if losses else None
        return x


class CourseGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        method = config.get('method', 'mha')
        if width % config['heads']:
            raise ValueError('Width must divide evenly across heads.')
        if method not in ('mha', 'moe', 'mla', 'dsa', 'linear', 'delta_hybrid', 'mhc'):
            raise ValueError(f'Unknown course method: {method}')
        self.token = nn.Embedding(config['vocab'], width)
        self.drop_emb = nn.Dropout(config.get('dropout', 0.))
        self.blocks = nn.ModuleList(
            CourseBlock(config, 'delta' if method == 'delta_hybrid' and layer == config['depth']-1
                        else method if method in ('mla', 'dsa', 'linear') else 'mha')
            for layer in range(config['depth']))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        if method == 'mhc':
            self.stream_bias = nn.Parameter(torch.randn(2, width) * .02)
            self.mix_logits = nn.Parameter(torch.eye(2).repeat(config['depth'], 1, 1) * 2)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        if method == 'delta_hybrid':
            nn.init.constant_(self.blocks[-1].attention.alpha.bias, 4.)
        self.aux_loss = None

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def set_training_step(self, step):
        for block in self.blocks:
            block.attention.training_step = step

    @staticmethod
    def _sinkhorn(logits):
        matrix = (logits - logits.amax()).exp()
        for _ in range(8):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True)
        return matrix

    def features(self, ids):
        x = self.drop_emb(self.token(ids))
        losses = []
        if self.config.get('method') == 'mhc':
            batch, length, width = x.shape
            x = x.unsqueeze(2) + self.stream_bias[None, None]
            for layer, block in enumerate(self.blocks):
                flat = x.permute(0, 2, 1, 3).reshape(batch * 2, length, width)
                updated = block(flat).reshape(batch, 2, length, width).permute(0, 2, 1, 3)
                matrix = self._sinkhorn(self.mix_logits[layer])
                x = torch.einsum('ij,btjd->btid', matrix, x) + (updated - x)
            x = x.mean(dim=2)
        else:
            for block in self.blocks:
                x = block(x)
                if block.aux_loss is not None:
                    losses.append(block.aux_loss)
        self.aux_loss = sum(losses) if losses else None
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return CourseGPT(config)
