# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import pvlib
from pathlib import Path
import warnings

# 确保导入你的模型结构定义
try:
    from step0_model import TransformerNBEATS
except ImportError:
    print("错误: 找不到 model.py，请确保文件在同一目录下。")
    exit()

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
BASE_DIR = Path("outputs/clean")
DATA_FILE = BASE_DIR / "dataset_ready_for_research_15min.csv"
MODEL_PATH = "transformer_best.pth"     # 训练好的模型权重
SCALER_PATH = "scaler.pkl"              # 训练集上fit好的归一化器
PARAM_FILE = BASE_DIR / "physics_params.csv" # 物理参数

RESULT_CSV = BASE_DIR / "final_inference_result.csv"
RESULT_IMG = BASE_DIR / "final_inference_plot.png"

# 【用户指定】想要查看的日期 (当地时间)
# 建议选择测试集范围内的时间 (最后15%)
PREFERRED_DATE_STR = "2019-10-15" 

# 参数必须与训练时完全一致
SEQ_LEN = 96          
PRED_LEN = 96         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P_RATED = 30.1        

# 站点参数
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE_MODEL = "UTC"            # 模型只认 UTC
TIMEZONE_LOCAL = "America/Los_Angeles" # 绘图给人看

RAW_COLS = [
    'Target_Power',
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI',
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 'NWP_Cloud', 'NWP_Precip'
]
# ===========================================

class PhysicalUncertainty:
    """物理引导的不确定性生成器"""
    def __init__(self, param_file):
        if not param_file.exists():
            print(f"[警告] 物理参数文件 {param_file} 不存在，使用默认宽松参数。")
            self.a, self.beta, self.c = 0.2, -0.5, 0.05
        else:
            df = pd.read_csv(param_file)
            params = dict(zip(df['parameter'], df['value']))
            self.a = params['a']
            self.beta = params['beta']
            self.c = params['c']
            print(f"已加载物理参数: a={self.a:.4f}, beta={self.beta:.4f}, c={self.c:.4f}")

    def get_sigma(self, p_pred_kw):
        """
        输入: 预测功率 (kW)
        输出: 动态标准差 sigma (kW)
        公式: sigma = (a * P_pu^beta + c) * P_pred
        """
        # 1. 转为标幺值 p.u.
        p_safe = np.maximum(p_pred_kw, 0.0)
        p_pu = p_safe / P_RATED
        
        # 2. 物理截断：避免分母为0，设定一个极小值
        p_pu_clamped = np.maximum(p_pu, 0.01)
        
        # 3. 计算相对波动率 I_F
        i_f = self.a * np.power(p_pu_clamped, self.beta) + self.c
        
        # 4. 还原绝对波动量
        sigma = i_f * p_safe
        return sigma

def add_solar_features(df):
    """
    与训练时完全一致的特征工程
    """
    print("   [预处理] 生成太阳几何特征...")
    # 强制 UTC
    try:
        times_utc = pd.to_datetime(df.index, utc=True)
    except:
        times_utc = pd.to_datetime(df.index).tz_localize('UTC')
        
    df.index = times_utc # 确保索引是 UTC
    
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE_MODEL)
    solpos = loc.get_solarposition(df.index)
    
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation']))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth']))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth']))
    return df

def load_environment():
    """加载模型、Scaler和数据，不进行任何 Fit 操作"""
    print("--- 正在加载环境 ---")
    
    # 1. 加载数据
    if not DATA_FILE.exists(): raise FileNotFoundError(f"{DATA_FILE} 未找到")
    df = pd.read_csv(DATA_FILE, index_col=0)
    df = add_solar_features(df)
    
    FULL_COLS = RAW_COLS + ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
    
    # 2. 加载 Scaler (核心：防泄露)
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError("Scaler 未找到！请先运行训练脚本。")
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler 已加载 (使用训练集参数)")
    
    # 3. 数据转换
    data_raw = df[FULL_COLS].values
    data_raw = np.nan_to_num(data_raw, nan=0.0)
    data_scaled = scaler.transform(data_raw) # Transform only
    
    # 4. 加载模型
    input_dim = len(FULL_COLS)
    model = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=input_dim,
        d_model=32, nhead=4, num_encoder_layers=2,
        dim_feedforward=128, input_seq_len=SEQ_LEN, output_len=PRED_LEN,
        dropout=0.3
    ).to(DEVICE)
    
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("模型权重未找到！")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 模型权重已加载")
    
    return model, scaler, data_scaled, df.index, input_dim

def predict_step(model, input_seq):
    """单步预测辅助函数"""
    tensor_x = torch.FloatTensor(input_seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor_x)
        # 形状修复
        if pred.dim() == 3: pred = pred.squeeze(-1)
    return pred.cpu().numpy().flatten()

def inverse_transform_target(y_scaled, scaler, input_dim):
    """只反归一化目标列 (假设第0列是功率)"""
    dummy = np.zeros((len(y_scaled), input_dim))
    dummy[:, 0] = y_scaled 
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

def apply_night_mask(df_in):
    """物理后处理：夜间置零"""
    df = df_in.copy()
    
    # 1. 提取时间列
    times = df['Time'] 
    
    # 2. 计算太阳高度角
    # 注意：pvlib 需要由 pandas Timestamp 组成的 Series 或 DatetimeIndex
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=df['Time'].dt.tz)
    solpos = loc.get_solarposition(times)
    
    # 3. 生成夜间掩码 (核心修复点：加上 .values 或 .to_numpy())
    # 只要太阳高度角 < -5度，就强制置0
    # .values 将其转换为纯布尔数组，忽略索引，避免与 df 的 RangeIndex 冲突
    is_night = (solpos['elevation'] < -5).values 
    
    cols_to_zero = ['Pred_24h', 'Lower_24h', 'Upper_24h', 'Pred_4h', 'Lower_4h', 'Upper_4h']
    
    for col in cols_to_zero:
        if col in df.columns:
            # 现在 is_night 是 numpy array，不会报错了
            df.loc[is_night, col] = 0.0
            
    return df

def run_inference():
    # 1. 加载所有组件
    model, scaler, data_scaled, time_index_utc, input_dim = load_environment()
    phys_engine = PhysicalUncertainty(PARAM_FILE)
    
    # 2. 定位测试集 (Test Set)
    # 我们之前的划分是 70/15/15。所以测试集是从 85% 开始。
    total_len = len(data_scaled)
    test_start_idx = int(total_len * 0.85)
    
    print(f"\n[数据集信息]")
    print(f"   总量: {total_len}")
    print(f"   测试集起点索引: {test_start_idx}")
    print(f"   测试集起点时间 (UTC): {time_index_utc[test_start_idx]}")
    
    # 3. 寻找用户指定日期的索引
    # 为了匹配方便，创建一个临时的 Local Time 索引
    time_index_local = time_index_utc.tz_convert(TIMEZONE_LOCAL)
    
    # 找到当天日期的所有索引
    target_date_str = pd.Timestamp(PREFERRED_DATE_STR).date().strftime('%Y-%m-%d')
    day_mask = (time_index_local.strftime('%Y-%m-%d') == target_date_str)
    indices = np.where(day_mask)[0]
    
    if len(indices) == 0:
        print(f"❌ 错误: 数据集中找不到日期 {PREFERRED_DATE_STR}")
        return
        
    # 取当天的起始点，作为预测起点
    # 注意：我们需要往前取 SEQ_LEN 个点作为输入
    start_idx = indices[0]
    
    # === 防泄露检查 ===
    if start_idx < test_start_idx:
        print(f"\n⚠️ 警告: 您选择的日期 {PREFERRED_DATE_STR} 位于 训练集/验证集 区域！")
        print("   为了展示论文结果，建议选择测试集（即 2019-10-xx 之后）的日期。")
        print("   (程序将继续执行，但请注意结论的严谨性)")
    else:
        print(f"\n✅ 确认: 日期 {PREFERRED_DATE_STR} 位于独立的测试集区域。")
    
    # 确保索引不越界
    if start_idx < SEQ_LEN or start_idx + PRED_LEN > total_len:
        print("❌ 错误: 索引越界 (数据不足以构建序列)")
        return

    # 4. 执行预测 (基于 UTC 数据)
    # ---------------------------------------------------------
    # 场景 A: 24h 日前预测 (Day-Ahead)
    # 一次性预测未来 96 个点
    input_seq_24h = data_scaled[start_idx - SEQ_LEN : start_idx]
    pred_24h_scaled = predict_step(model, input_seq_24h)
    pred_24h_real = inverse_transform_target(pred_24h_scaled, scaler, input_dim)
    
    # 计算物理区间
    sigma_24h = phys_engine.get_sigma(pred_24h_real)
    lower_24h = pred_24h_real - 1.96 * sigma_24h
    upper_24h = pred_24h_real + 1.96 * sigma_24h
    
    # ---------------------------------------------------------
    # 场景 B: 4h 滚动预测 (Intra-Day Rolling)
    # 每隔 4小时 (16个点) 刷新一次预测
    pred_4h_real = []
    upper_4h_real = [] # 这里只存上界演示
    
    step_size = 16 # 4 hours * 4 points/hour
    
    # 循环填满 96 个点
    for i in range(0, PRED_LEN, step_size):
        curr_idx = start_idx + i
        input_seq_roll = data_scaled[curr_idx - SEQ_LEN : curr_idx]
        
        # 预测
        chunk_pred_scaled = predict_step(model, input_seq_roll)
        chunk_pred_real = inverse_transform_target(chunk_pred_scaled, scaler, input_dim)
        
        # 物理区间
        chunk_sigma = phys_engine.get_sigma(chunk_pred_real)
        chunk_upper = chunk_pred_real + 1.96 * chunk_sigma
        
        # 截取前 4 小时有效
        valid_len = min(step_size, PRED_LEN - i)
        pred_4h_real.extend(chunk_pred_real[:valid_len])
        upper_4h_real.extend(chunk_upper[:valid_len])
        
    pred_4h_real = np.array(pred_4h_real)
    upper_4h_real = np.array(upper_4h_real)
    
    # 反推 sigma 用于绘图 (简化)
    sigma_4h_derived = (upper_4h_real - pred_4h_real) / 1.96
    lower_4h_real = pred_4h_real - 1.96 * sigma_4h_derived
    
    # 5. 获取 Ground Truth
    gt_scaled = data_scaled[start_idx : start_idx + PRED_LEN, 0]
    gt_real = inverse_transform_target(gt_scaled, scaler, input_dim)
    
    # 6. 整合结果
    timestamps = time_index_local[start_idx : start_idx + PRED_LEN]
    
    df_res = pd.DataFrame({
        'Time': timestamps,
        'Actual': gt_real,
        'Pred_24h': np.maximum(pred_24h_real, 0),
        'Lower_24h': np.maximum(lower_24h, 0),
        'Upper_24h': np.maximum(upper_24h, 0),
        'Pred_4h': np.maximum(pred_4h_real, 0),
        'Lower_4h': np.maximum(lower_4h_real, 0),
        'Upper_4h': np.maximum(upper_4h_real, 0)
    })
    
    # 夜间修正
    df_res = apply_night_mask(df_res)
    
    # 保存 CSV
    df_res.to_csv(RESULT_CSV, index=False)
    print(f"CSV 保存成功: {RESULT_CSV}")
    
    # 7. 绘图 (Paper Quality)
    plot_results(df_res)

def plot_results(df):
    """绘制出版级对比图"""
    plt.figure(figsize=(12, 6))
    
    # X轴: 去掉时区信息，只留时间，避免 matplotlib 混乱
    times = df['Time'].dt.tz_localize(None)
    
    # 1. 真实值
    plt.plot(times, df['Actual'], color='black', linewidth=2, label='Actual Power')
    
    # 2. 24h 预测 (蓝色)
    plt.plot(times, df['Pred_24h'], color='dodgerblue', linestyle='--', linewidth=2, label='DA Forecast (24h)')
    plt.fill_between(times, df['Lower_24h'], df['Upper_24h'], color='dodgerblue', alpha=0.15, label='95% PI (Physics-Guided)')
    
    # 3. 4h 预测 (红色) - 可选，如果不想要太乱可以注释掉
    plt.plot(times, df['Pred_4h'], color='crimson', linestyle='-', linewidth=1.5, alpha=0.8, label='ID Forecast (4h Rolling)')
    # plt.fill_between(times, df['Lower_4h'], df['Upper_4h'], color='crimson', alpha=0.05)
    
    # 格式化
    plt.title(f"Physics-Informed Forecasting: {df['Time'].iloc[0].date()} (Test Set)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=12)
    plt.xlabel("Local Time", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.ylim(bottom=-0.5) # 稍微留点底
    
    # 时间格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(RESULT_IMG, dpi=300)
    print(f"图片保存成功: {RESULT_IMG}")
    # plt.show()

if __name__ == "__main__":
    run_inference()