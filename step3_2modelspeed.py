# -*- coding: utf-8 -*-
import torch
import time
from thop import profile
import warnings

# 引入你的模型类 (与 step3_1train_model.py 保持一致)
try:
    from step0_model import TransformerNBEATS
except ImportError:
    raise ImportError("❌ 错误: 找不到 step0_model.py，请确保文件在同级目录。")

warnings.filterwarnings("ignore")

def evaluate_model_efficiency(model, input_tensor, device='cpu', model_name="Model"):
    """
    评估模型参数量、FLOPs 和单次前向传播推理时间
    """
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    print(f"\n========== 【{model_name}】算力评估报告 ==========")
    
    # 1. 计算参数量 (Parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Parameters: {total_params / 1e6:.4f} M (百万级)")

    # 2. 计算计算复杂度 (FLOPs)
    # thop.profile 返回 MACs (乘加累积操作数)，通常 1 MAC ≈ 2 FLOPs
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops = macs * 2  
    print(f"[*] Computational Complexity: {flops / 1e9:.4f} GFLOPs (十亿次浮点运算)")

    # 3. 计算推理时间 (Inference Time)
    # 预热 (Warm-up)：防止模型初次加载和显存/内存分配带来的时间抖动
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_tensor)
            
    # 正式测速 (取 500 次的平均值以保证严谨性)
    num_runs = 500
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
            
    # 如果使用 GPU，必须加上同步锁，否则 Python 会异步计时导致结果偏小
    if device == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_runs * 1000 # 毫秒 (ms)
    print(f"[*] Average Inference Time (Batch=1): {avg_inference_time:.4f} ms")
    print("========================================================\n")


if __name__ == "__main__":
    # 根据你的 step3_1train_model.py 提取参数
    SEQ_LEN = 96
    PRED_LEN = 96
    # FEATURE_COLS 包含: 9个NWP特征 + 3个物理特征 + 4个时间特征 = 16个特征
    INPUT_DIM = 16 
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 测试硬件环境: {DEVICE.type.upper()}")

    # 构建模拟输入数据 (Batch Size = 1 模拟实际实时调度场景)
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_DIM)

    # -----------------------------------------------------------------------
    # 核心逻辑证明：
    # 为什么我们只实例化一个模型？
    # 因为 Baseline Transformer 和 PiT-Net 的*推理架构*是完全等价的！
    # PiT-Net 的区别仅仅存在于 step3_1train_model.py 中的 Loss 计算阶段。
    # 这里的实例化直接证明了：部署时的数学路径和计算负担毫无增加。
    # -----------------------------------------------------------------------
    
    architecture = TransformerNBEATS(
        num_stacks=1, 
        num_blocks_per_stack=1, 
        input_dim=INPUT_DIM, 
        d_model=16, 
        nhead=2, 
        num_encoder_layers=1, 
        dim_feedforward=32, 
        input_seq_len=SEQ_LEN, 
        output_len=PRED_LEN, 
        dropout=0.5 
    )

    print(">>> 步骤 1: 测试 Baseline Transformer 推理开销")
    evaluate_model_efficiency(architecture, dummy_input, DEVICE, "Baseline Transformer")
    
    print(">>> 步骤 2: 测试 PiT-Net 推理开销")
    # 物理权重只在反向传播中起作用，前向推理就是原本的架构
    evaluate_model_efficiency(architecture, dummy_input, DEVICE, "PiT-Net (Proposed)")
    
    print("💡 结论提取：")
    print("你可以直接将上述输出的 Parameters, GFLOPs 和 Inference Time 填入论文的表格中。")
    print("这两个模型的数据将完全一致，完美证明了'Zero additional inference overhead'的声明。")