# -*- coding: utf-8 -*-

"""
版本: V2.0 (Decoupled Architecture)
名称: Transformer-NBEATS Pure Point Forecaster

[核心设计思想 - 物理数据解耦]
此模型仅负责 "Accuracy" (高精度点预测)。
它利用 Transformer 的注意力机制和 N-BEATS 的残差分解能力，
从历史功率和 NWP 天气数据中学习映射关系 f(X) -> Y_hat。

不确定性量化 (Reliability) 将在模型外部，
由 "Scale-Invariant Physical Power Law" (尺度不变物理幂律) 模块统一处理。
"""

import torch
import torch.nn as nn
import math

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
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe.squeeze(1)[:x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder_proj = nn.Linear(d_model, input_dim) 

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # Transformer expects: [seq_len, batch_size, d_model]
        x_t = x.transpose(0, 1) 
        x_proj = self.input_proj(x_t)
        x_pos = self.pos_encoder(x_proj)
        output = self.transformer_encoder(x_pos)
        output = self.decoder_proj(output)
        # Return to: [batch_size, seq_len, input_dim]
        return output.transpose(0, 1)

class NBEATSBlock(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, 
                 input_seq_len, output_len, dropout=0.1):
        super(NBEATSBlock, self).__init__()
        
        # 1. Feature Extraction (Transformer)
        self.feature_extractor = TransformerBlock(input_dim, d_model, nhead, 
                                                  num_encoder_layers, dim_feedforward, dropout)
        
        # Flatten dimension for Fully Connected layers
        flatten_dim = input_seq_len * input_dim
        
        # 2. Theta Estimator (predicts coefficients for basis functions)
        # output_len for forecast, input_seq_len * input_dim for backcast
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
        # x: [batch, seq_len, input_dim]
        
        # A. Extract Features
        features = self.feature_extractor(x)
        
        # B. Estimate Thetas
        thetas = self.theta_fc(features)
        
        # Split thetas into backcast (reconstruction) and forecast parts
        theta_b = thetas[:, :self.input_seq_len * self.input_dim]
        theta_f = thetas[:, self.input_seq_len * self.input_dim:]
        
        # C. Generate Backcast (fitting historical curve)
        backcast = theta_b.view(-1, self.input_seq_len, self.input_dim)
        
        # D. Generate Forecast (predicting future curve)
        # Note: In pure N-BEATS, this uses basis functions. 
        # Here we simplify to direct regression for the Transformer variant.
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
        residual = x # Initialize residual with input
        
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            # Residual calculation: remove the signal explained by this block
            residual = residual - backcast
            # Accumulate forecast
            stack_forecast = stack_forecast + block_forecast
            
        return stack_forecast, residual

class TransformerNBEATS(nn.Module):
    """
    Main Model Class: Decoupled Architecture
    Input: Historical PV + NWP Weather Features
    Output: Pure Point Forecast (Power)
    """
    def __init__(self, num_stacks, num_blocks_per_stack, input_dim, d_model, nhead, 
                 num_encoder_layers, dim_feedforward, input_seq_len, output_len, dropout=0.1):
        super(TransformerNBEATS, self).__init__()
        
        self.output_len = output_len
        
        # Stack of N-BEATS blocks
        self.nbeats_stacks = nn.ModuleList(
            [NBEATSStack(num_blocks_per_stack, input_dim, d_model, nhead, num_encoder_layers, 
                         dim_feedforward, input_seq_len, output_len, dropout) 
             for _ in range(num_stacks)]
        )

    def forward(self, x):
        # x: [batch_size, input_seq_len, input_dim]
        
        final_forecast = torch.zeros(x.size(0), self.output_len, device=x.device)
        current_input = x
        
        # Doubly Residual Stacking
        for stack in self.nbeats_stacks:
            stack_forecast, residual_from_stack = stack(current_input)
            final_forecast = final_forecast + stack_forecast
            current_input = residual_from_stack
            
        # Return only the point forecast
        # The uncertainty intervals will be generated externally using the Power Law.
        return final_forecast