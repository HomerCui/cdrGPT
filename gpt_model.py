import torch
import torch.nn as nn
import numpy as np
import json
import os


"""
该程序主要用来描述网络结构

torch.set_printoptions(profile="full")
"""

# 超参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device is {device}")

n_heads = 8  # 头数
d_k = d_v = 64  # 注意力矩阵尺寸
vs = [0]  # 模拟静态变量，vocab_size，词典中的总token数量
# vocab_size = len(dic_0)  # 超参数：字典长度，用于做embedding前的维度和softmax时的维度
smi_len = padding = padding_size = max_smi_len = 16  # 最大接收长度，padding长度
model_d = 512  # 嵌入维度
batch_size = 512
n_blocks = 8  # 块数

# 载入字典，获取字典中的token数量
d_path = "./dictionary/dic_c_1.json"
if os.path.exists(d_path):
    with open(d_path, 'r') as d_file:
        d_count = json.load(d_file)
    vs[0] = len(d_count)  # 将总词数装入变量
vocab_size = vs[0]
print(f"get vocab_size as {vocab_size}")


def pad_mask_metrix(s_q, s_k):
    """
    生成当前用来mask padding位的矩阵
    :param s_q:当前操作的输入
    :param s_k:
    :return:将padding位标记为mask的矩阵[batch_size, smiles_len, smiles_len]
    """
    batch_size, len_q = s_q.size()
    batch_size, len_k = s_k.size()
    pad_mask = s_k.data.eq(0).unsqueeze(1).to(device)
    return pad_mask.expand(batch_size, len_q, len_k).to(device)


def step_mask_metrix(inputs):
    """
    多头注意力机制的掩码矩阵生成器
    :param inputs: 当前操作的数据，用来获得矩阵的尺寸
    :return: 将未来字符标记为mask的矩阵[batch_size, smiles_len, smiles_len]
    """
    shape = [inputs.size(0), inputs.size(1), inputs.size(1)]
    mask = torch.tril(torch.ones(size=shape, device=device)).byte()
    return mask.data.eq(0)


class WorldPositionEmbedding(nn.Module):
    # 用于做两个embedding，还包含了layer norm和dropout
    def __init__(self, vocab_size=vocab_size, smi_len=smi_len, model_d=model_d, drop=0.5):
        """
        :param vocab_size: 预计字典最大数量
        :param smi_len: padding后每个输入的分子长度
        :param model_d: 模型的输入维度
        :param drop: dropout预设为0.5，在0-1之间可调
        """
        super().__init__()
        self.world_embedding = nn.Embedding(vocab_size, model_d).to(device)
        self.position_embedding = nn.Embedding(2048, model_d).to(device)  # 实际上报错的地方应该是这里！！！！！！！！
        self.dropout = nn.Dropout(drop)

    def forward(self, input_label):
        """
        实现了world embedding和position embedding，之后进行了layer norm和dropout
        :param input_label: 输入的label或者input，类型是tensor，shape是[batch_size, smiles_len(padding_size)]
        :return:embedding后的数据，直接输入模型进行训练，shape是[batch_size, smiles_len(padding_size), model_dimension]
                同时还返回了1个矩阵用来mask输入内容
        """
        smile_len = input_label.size(1)  # 扩展维度使匹配
        # 这里position的分配方式要改变，使用传统的三角函数分配，回去参考一下GPT2的实现
        position = torch.arange(smile_len, dtype=torch.long, device=device)
        position = position.unsqueeze(0).expand_as(input_label.to(device))

        w_p_embedding = self.world_embedding(input_label) + self.position_embedding(position)
        # position embedding这里容易报错，考虑换个实现方式！！！！！！！！！！！！！！！！！！

        layer_norm = nn.LayerNorm(w_p_embedding.size(-1)).to(device)
        out = layer_norm(w_p_embedding.to(device))
        out_embedding = self.dropout(out)

        pad_mask = pad_mask_metrix(input_label.to(device), input_label.to(device)).to(device)
        step_mask = step_mask_metrix(input_label.to(device))
        mask = pad_mask + step_mask

        return out_embedding, mask


class FeedForward(nn.Module):

    def __init__(self, model_d):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(model_d, model_d * 4, bias=False),
            nn.ReLU(),
            nn.Linear(model_d * 4, model_d, bias=False),
        )
        self.layer_norm = nn.LayerNorm(model_d)

    def forward(self, input_data):
        """
        多头注意力后的第二个主要模块
        :param input_data:
        :return:
        """
        res = input_data.to(device)
        output = self.fcn(input_data.to(device))
        return self.layer_norm(output + res)


class MaskDPAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask):
        """
        进行掩码
        """
        attention_s = torch.matmul(q, k.transpose(-1, -2) / np.sqrt(d_k)).to(device)
        attention_s.masked_fill_(mask, -1e9).to(device)
        attention = nn.Softmax(dim=-1)(attention_s)  # [batch_size, n_heads, smiles_len, smiles_len]
        context_attention = torch.matmul(attention, v).to(device)  # [batch_size, n_heads, smiles_len, d_v]

        return attention, context_attention


class MHAttention(nn.Module):

    def __init__(self, model_d, bias=False):
        super().__init__()
        self.w_q = nn.Linear(model_d, n_heads * d_k, bias).to(device)
        self.w_k = nn.Linear(model_d, n_heads * d_k, bias).to(device)
        self.w_v = nn.Linear(model_d, n_heads * d_v, bias).to(device)
        self.fcn = nn.Linear(n_heads * d_v, model_d, bias).to(device)
        self.layer_norm = nn.LayerNorm(model_d).to(device)

    def forward(self, q, k, v, mask):
        """
        多头注意力，与掩码部分联用形成masked多头注意力，GPT的第一个主要模块
        :return:
        """
        batch_size = q.size(0)
        res = q
        q_out = self.w_q(q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_out = self.w_k(k).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_out = self.w_v(v).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        attention, context_attention = MaskDPAttention()(q_out, k_out, v_out, mask)
        context_attention = context_attention.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v)
        output = self.fcn(context_attention)

        output = self.layer_norm(output.to(device) + res.to(device))

        return output, attention


class GPTBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.self_attention = MHAttention(model_d=model_d)
        self.feed_forward = FeedForward(model_d=model_d)

    def forward(self, smiles, mask):
        output, attention = self.self_attention(smiles.to(device), smiles.to(device), smiles.to(device), mask.to(device))
        output = self.feed_forward(output.to(device))
        return output, attention  # [batch_size, smiles_len, model_d]  [batch_size, n_heads, smiles_len, smiles_len]


class Layers(nn.Module):
    
    def __init__(self, n_blocks=n_blocks):
        super().__init__()
        self.gpt_blocks = nn.ModuleList(
            [GPTBlock()for _ in range(n_blocks)]
            )
        self.embedding = WorldPositionEmbedding()
        
    def forward(self, inputs):
        output, mask =self.embedding(inputs.to(device))
        self_attentions =[]
        for block in self.gpt_blocks:
            output, attention = block(output,mask)
            self_attentions.append(attention)
            
        return output, self_attentions


class CovProjection(nn.Module):

    def __init__(self, model_d=model_d, vocab_size=vocab_size):
        super().__init__()
        self.lin_1 = nn.Linear(model_d, model_d * 4)
        self.relu = nn.ReLU()
        self.lin_2 = nn.Linear(model_d * 4, vocab_size)

    def forward(self, x):
        x = self.lin_1(x.to(device))
        x = self.relu(x.to(device))
        x = self.lin_2(x.to(device))

        return x


class ConvProjection(nn.Module):
    def __init__(self, model_d=model_d, vocab_size=vocab_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=model_d, out_channels=model_d * 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=model_d * 4, out_channels=512, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=vocab_size, kernel_size=1)

    def forward(self, x):
        res = x.to(device)
        x = x.permute(0, 2, 1).to(device)  # 将输入转换为卷积层所需的维度顺序
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)  # 将输出转换回原始维度顺序
        return x


class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = Layers()
        self.projection = ConvProjection()

    def forward(self, smiles):
        output, attention = self.layers(smiles.to(device))
        out = self.projection(output.to(device))
        torch.set_printoptions(profile="full")
        return out.reshape(-1, out.size(-1))
        # out.reshape(-1, out.size(-1)) [batch_size * smi_len, vocab_size]




"""
成于2024.4.7 国科大雁栖湖 2公寓
cwr
"""
