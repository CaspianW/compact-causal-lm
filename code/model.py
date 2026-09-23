"""课程提供的 baseline：从随机初始化开始训练的小型 GPT。"""

import torch  # 张量运算、设备管理和随机采样
from torch import nn  # 神经网络层和 Module 基类
from torch.nn import functional as F  # 注意力和 log_softmax 等函数


class Block(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()  # 初始化 nn.Module，注册后续定义的子层
        self.heads = heads  # 注意力头数量：把 width 拆成 heads 组并行计算

        self.norm1 = nn.LayerNorm(width)  # 注意力前的归一化层
        self.norm2 = nn.LayerNorm(width)  # MLP 前的归一化层

        # 一次线性变换同时生成 Q、K、V，所以输出维度是 3 * width。
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)  # 多个注意力头合并后的输出变换

        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),  # 先扩大隐藏维度，增加表达能力
            nn.GELU(),  # 非线性激活函数
            nn.Linear(4 * width, width),  # 把维度变回 width，方便残差相加
        )

    def forward(self, x):
        batch, length, width = x.shape  # x 的形状：[批次大小, 序列长度, 特征维度]

        normalized = self.norm1(x)  # 先对每个 token 的特征做 LayerNorm
        qkv = self.qkv(normalized)  # 每个 token 同时生成 query、key、value
        qkv = qkv.view(
            batch, length, 3, self.heads, width // self.heads
        )  # 拆成：[批次, 序列, QKV, 注意力头, 每头维度]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 分离 Q/K/V，并调整为注意力需要的顺序

        # 因果注意力：当前位置只能读取自己和前面的 token，不能偷看未来。
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.proj(attended)  # 残差连接：保留原输入并加上注意力结果

        # 第二个残差分支：归一化后经过 MLP，再加回 x。
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()  # 初始化 GPT 这个神经网络模块
        self.config = dict(config)  # 保存一份配置副本，便于记录实验信息
        self.context = config['context']  # 模型一次最多处理的 token 数
        width = config['width']  # 每个 token 的隐藏向量维度

        # token embedding：把 token 编号转换成 width 维向量。
        self.token = nn.Embedding(config['vocab'], width)
        # position embedding：为序列中的每个位置学习一个 width 维向量。
        self.pos = nn.Embedding(self.context, width)

        # 堆叠 depth 个 Transformer block。
        self.blocks = nn.ModuleList(
            [Block(width, config['heads']) for _ in range(config['depth'])]
        )
        self.norm = nn.LayerNorm(width)  # 所有 block 之后的最终归一化
        self.head = nn.Linear(width, config['vocab'], bias=False)  # hidden vector -> 每个 token 的 logits

        self.apply(self.initialize)  # 递归初始化所有 Linear 和 Embedding 的权重
        self.head.weight = self.token.weight  # 权重共享：输入 embedding 和输出分类器使用同一矩阵

    @staticmethod
    def initialize(module):
        # 对 Linear 和 Embedding 的权重使用均值 0、标准差 .02 的正态分布初始化。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)  # 如果该层有 bias，就初始化为 0

    def features(self, ids):
        length = ids.shape[1]  # 当前输入序列长度
        positions = torch.arange(length, device=ids.device)  # 位置编号：[0, 1, ..., length-1]
        x = self.token(ids)  # token 编号 -> [batch, length, width]
        x = x + self.pos(positions)  # 加上位置信息；[length, width] 会广播到每个样本

        for block in self.blocks:
            x = block(x)  # 依次通过每个 Transformer block
        return self.norm(x)  # 返回最终 hidden features

    def forward(self, ids):
        """训练接口：返回未归一化的 next-token logits。"""
        return self.head(self.features(ids))  # [batch, length, width] -> [batch, length, vocab]

    def predict_log_probs(self, ids):
        """评估接口：返回每个位置对整个词表的归一化 log 概率。"""
        return F.log_softmax(self(ids).float(), dim=-1)  # 沿词表维度转换为 log probabilities


def build_model(config):
    """供 train.py 和 evaluate.py 创建模型。"""
    return GPT(config)
