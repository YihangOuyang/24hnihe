# -*- coding: utf-8 -*-

"""
版本: V2.1 (Physics-Informed Loss Integrated)
名称: Transformer-NBEATS Point Forecaster with Physics Loss

[核心创新 - 物理加权损失函数]
本模块新增了 PhysicsWeightedMSELoss 类。
该损失函数利用 "波动缩放律" (Fluctuation Scaling Law) I_F = a * P^beta + c，
动态调整每个样本的梯度权重：
1. 高功率/稳态 (Clear Sky): I_F 小 -> 权重高 -> 强迫模型精准拟合
2. 低功率/瞬态 (Cloudy/Morning): I_F 大 -> 权重低 -> 允许模型存在误差
"""

import torch
import torch.nn as nn
import math

# ==========================================
# 1. 基础组件 (Positional Encoding)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.squeeze(1)[:x.size(1), :]
        return x

# ==========================================
# 2. Transformer 模块
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder_proj = nn.Linear(d_model, input_dim) 

    def forward(self, x):
        x_t = x.transpose(0, 1) 
        x_proj = self.input_proj(x_t)
        x_pos = self.pos_encoder(x_proj)
        output = self.transformer_encoder(x_pos)
        output = self.decoder_proj(output)
        return output.transpose(0, 1)

# ==========================================
# 3. N-BEATS 模块
# ==========================================
class NBEATSBlock(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, 
                 input_seq_len, output_len, dropout=0.1):
        super(NBEATSBlock, self).__init__()
        self.feature_extractor = TransformerBlock(input_dim, d_model, nhead, 
                                                  num_encoder_layers, dim_feedforward, dropout)
        flatten_dim = input_seq_len * input_dim
        self.theta_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_len + flatten_dim) 
        )
        self.input_seq_len = input_seq_len
        self.input_dim = input_dim
        self.output_len = output_len

    def forward(self, x):
        features = self.feature_extractor(x)
        thetas = self.theta_fc(features)
        theta_b = thetas[:, :self.input_seq_len * self.input_dim]
        theta_f = thetas[:, self.input_seq_len * self.input_dim:]
        backcast = theta_b.view(-1, self.input_seq_len, self.input_dim)
        forecast = theta_f 
        return backcast, forecast

class NBEATSStack(nn.Module):
    def __init__(self, num_blocks, input_dim, d_model, nhead, num_encoder_layers, 
                 dim_feedforward, input_seq_len, output_len, dropout=0.1):
        super(NBEATSStack, self).__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_dim, d_model, nhead, num_encoder_layers, 
                        dim_feedforward, input_seq_len, output_len, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        stack_forecast = torch.zeros(x.size(0), self.blocks[0].output_len, device=x.device)
        residual = x 
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            stack_forecast = stack_forecast + block_forecast
        return stack_forecast, residual

# ==========================================
# 4. 主模型架构
# ==========================================
class TransformerNBEATS(nn.Module):
    def __init__(self, num_stacks, num_blocks_per_stack, input_dim, d_model, nhead, 
                 num_encoder_layers, dim_feedforward, input_seq_len, output_len, dropout=0.1):
        super(TransformerNBEATS, self).__init__()
        self.output_len = output_len
        self.nbeats_stacks = nn.ModuleList(
            [NBEATSStack(num_blocks_per_stack, input_dim, d_model, nhead, num_encoder_layers, 
                         dim_feedforward, input_seq_len, output_len, dropout) 
             for _ in range(num_stacks)]
        )

    def forward(self, x):
        final_forecast = torch.zeros(x.size(0), self.output_len, device=x.device)
        current_input = x
        for stack in self.nbeats_stacks:
            stack_forecast, residual_from_stack = stack(current_input)
            final_forecast = final_forecast + stack_forecast
            current_input = residual_from_stack
        return final_forecast

# ==============================================================================
# 5. [核心创新] 物理加权损失函数 (Physically Weighted Loss)
# ==============================================================================
class PhysicsWeightedMSELoss(nn.Module):
    """
    创新点: 基于 'Fluctuation Scaling Law' (波动缩放律) 的自适应损失函数
    公式: Loss = mean( Weight * (y_pred - y_true)^2 )
    其中 Weight ~ 1 / I_F (波动越小，权重越大)
    I_F = a * (P_clearsky)^beta + c
    """
    def __init__(self, a, beta, c, capacity=1.0, epsilon=1e-4):
        """
        Args:
            a, beta, c: Step 2 拟合得到的物理参数 (a*P^beta + c)
            capacity: 电站额定容量 (用于归一化 P_clearsky)
            epsilon: 防止除零的微小量
        """
        super(PhysicsWeightedMSELoss, self).__init__()
        self.a = a
        self.beta = beta
        self.c = c
        self.capacity = capacity
        self.epsilon = epsilon
        self.base_loss = nn.MSELoss(reduction='none') # 不进行平均，保留每个样本的 Loss

    def forward(self, pred, target, physics_proxy):
        """
        Args:
            pred: 模型预测值 [Batch, Seq]
            target: 真实值 [Batch, Seq]
            physics_proxy: 物理基准值 [Batch, Seq] 
                           (通常是 'P_clearsky' 或 'Solar_Elevation' 映射出的理论功率)
        """
        # 1. 计算基础 MSE (逐元素)
        raw_mse = self.base_loss(pred, target)
        
        # 2. 计算物理基准的标幺值 (p.u.)
        # physics_proxy 应该是 Clearsky Power。如果是 Solar_El，需要在外部转一下。
        p_pu = physics_proxy / self.capacity
        p_pu = torch.clamp(p_pu, min=0.01, max=1.0) # 物理截断，防止 beta 为负数时爆炸
        
        # 3. 计算预期波动强度 I_F (Fluctuation Intensity)
        # 根据你的发现：I_F = a * P^beta + c
        # P 越大，I_F 越小 (因为 beta < 0)
        expected_volatility = self.a * torch.pow(p_pu, self.beta) + self.c
        
        # 4. 计算权重 (Inverted Volatility)
        # 波动越小(Clear Sky)，我们希望惩罚越重 -> 权重越大
        # 波动越大(Cloudy)，我们允许误差 -> 权重越小
        raw_weights = 1.0 / (expected_volatility + self.epsilon)
        raw_weights = torch.clamp(raw_weights, max=10.0)
        # 5. 权重归一化 (关键步骤！)
        # 保持一个 Batch 内的平均权重为 1，防止梯度消失或爆炸
        # 这样只改变梯度的"方向"（关注点），不改变梯度的"大小"
        weights_norm = raw_weights / (raw_weights.mean() + 1e-8)
        
        # 6. 加权 Loss
        weighted_loss = (raw_mse * weights_norm).mean()
        
        return weighted_loss