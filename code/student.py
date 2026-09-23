"""学生模型入口：先实现一个可测试的 GPT 版本。"""

import torch  # 张量运算
from torch import nn  # 神经网络层
from torch.nn import functional as F  # log_softmax


class Block(nn.Module):
    """一个带因果自注意力和 MLP 的 Transformer block。"""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x):
        # 上三角为 True，禁止当前位置读取未来 token。
        length = x.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        normalized = self.norm1(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    """符合项目接口的学生 GPT。"""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        vocab = config['vocab']

        self.token = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            [Block(width, config['heads']) for _ in range(config['depth'])]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        # 使用与 baseline 相近的初始化，便于公平比较。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        x = self.token(ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """训练接口：返回 [batch, time, vocab] 的未归一化 logits。"""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """评估接口：返回归一化的 log probabilities。"""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    # train.py 和 evaluate.py 都通过这个函数创建模型。
    return GPT(config)
