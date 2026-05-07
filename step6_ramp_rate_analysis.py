# -*- coding: utf-8 -*-
"""
Step 7: Daily Cumulative Energy Yield Analysis
评估不同模型在不同天气下，对“全天总发电量(kWh)”的预测准确率。
包含所有基线模型与提出模型的全量对比。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ================= 配置路径 =================
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_PATH = CURRENT_DIR / "outputs" / "benchmark" / "benchmark_all_predictions_timeseries.csv"

OUT_DIR_FIG = CURRENT_DIR / "outputs" / "paper_figures"
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
OUT_DIR_ORIGIN = CURRENT_DIR / "outputs" / "origin_data"
OUT_DIR_ORIGIN.mkdir(parents=True, exist_ok=True)

print(">>> 正在读取预测结果数据...")
df = pd.read_csv(DATA_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df['Date'] = df.index.date

# 过滤掉夜间零值，只保留有效发电时段进行天气打标
df_day = df[df['Actual_Power_Fixed'] > 0.1].copy()

# ================= 1. 严格复用之前的天气打标逻辑 =================
print(">>> 正在进行天气特征聚类打标 (保持严格一致)...")
daily_stats = df_day.groupby('Date')['Actual_Power_Fixed'].agg(
    mean_power='mean',
    volatility=lambda x: np.std(np.diff(x)) if len(x) > 1 else 0
)

vol_threshold = daily_stats['volatility'].quantile(0.66)
mean_threshold = daily_stats[daily_stats['volatility'] <= vol_threshold]['mean_power'].median()

def classify_regime(row):
    if row['volatility'] > vol_threshold:
        return 'Partly Cloudy'
    elif row['mean_power'] > mean_threshold:
        return 'Clear Sky'
    else:
        return 'Overcast'

daily_stats['Regime'] = daily_stats.apply(classify_regime, axis=1)

# ================= 2. 计算日累计发电量 (Cumulative Energy Yield) =================
# 功率 (kW) * 时间间隔 (15分钟 = 0.25小时) = 能量 (kWh)
print(">>> 正在计算每日总发电量及相对误差...")

power_cols = ['Actual_Power_Fixed'] + [c for c in df.columns if c.startswith('Pred_')]
# 在包含夜间的全天数据上进行按天求和积分
df_daily_energy = df.groupby('Date')[power_cols].sum() * 0.25

# 合并天气标签
df_daily_energy = df_daily_energy.merge(daily_stats[['Regime']], left_index=True, right_index=True)

# 计算每个模型的日发电量绝对百分比误差 (MAPE, %)
models = [c for c in power_cols if c != 'Actual_Power_Fixed']
error_cols = []

for model in models:
    err_col_name = f"Err%_{model.replace('Pred_', '')}"
    df_daily_energy[err_col_name] = (
        np.abs(df_daily_energy[model] - df_daily_energy['Actual_Power_Fixed']) 
        / df_daily_energy['Actual_Power_Fixed'] * 100
    )
    error_cols.append(err_col_name)

# ================= 3. 导出 Origin 数据 =================
# 计算每个天气下，各个模型的“平均日电量误差率”
df_mean_error = df_daily_energy.groupby('Regime')[error_cols].mean().reset_index()

# 整理列名，区分你的模型和基线
rename_dict = {}
for c in error_cols:
    model_name = c.replace('Err%_', '')
    if model_name == 'Transformer':
        rename_dict[c] = 'Transformer(Baseline)'
    elif model_name == 'PiT-Net':
        rename_dict[c] = 'PiT-Net(Proposed)'
    else:
        rename_dict[c] = model_name

df_mean_error.rename(columns=rename_dict, inplace=True)

# 对天气类别进行排序
df_mean_error['Regime'] = pd.Categorical(df_mean_error['Regime'], categories=['Clear Sky', 'Overcast', 'Partly Cloudy'], ordered=True)
df_mean_error.sort_values('Regime', inplace=True)

origin_csv_path = OUT_DIR_ORIGIN / "origin_daily_yield_mape.csv"
df_mean_error.to_csv(origin_csv_path, index=False)
print(f"✅ Origin 日电量误差柱状图数据已保存至: {origin_csv_path.name}")

# ================= 4. 生成 Python 预览图 (全模型柱状图) =================
print(">>> 正在生成全模型对比预览图...")
df_melt = df_mean_error.melt(id_vars=['Regime'], var_name='Model', value_name='Mean Daily Yield Error (%)')

# 为所有 8 个模型设置专属颜色，PiT-Net 设为最醒目的红色
full_palette = {
    'Ridge': '#7f7f7f',               # 灰色
    'RandomForest': '#8c564b',        # 棕色
    'XGBoost': '#2ca02c',             # 绿色
    'MLP': '#e377c2',                 # 粉色
    'LSTM': '#1f77b4',                # 蓝色
    'CNN': '#ff7f0e',                 # 橙色
    'Transformer(Baseline)': '#9467bd',# 紫色
    'PiT-Net(Proposed)': '#d62728'    # 红色
}

# 强制图例和柱状图的排列顺序，把你的模型放在最右侧压轴
model_order = ['Ridge', 'RandomForest', 'XGBoost', 'MLP', 'LSTM', 'CNN', 'Transformer(Baseline)', 'PiT-Net(Proposed)']

# 加宽画布以容纳 8 根柱子
plt.figure(figsize=(14, 6))
sns.barplot(data=df_melt, x='Regime', y='Mean Daily Yield Error (%)', hue='Model', 
            palette=full_palette, hue_order=model_order)

plt.title('Daily Cumulative Energy Yield Error by Weather Regime (All Models)', fontsize=15)
plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=13)
plt.xlabel('Weather Regime', fontsize=13)

# 调整图例位置，避免遮挡数据
plt.legend(title='Models', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

fig_save_path = OUT_DIR_FIG / "daily_yield_error_barplot_all.png"
plt.savefig(fig_save_path, dpi=300)
print(f"✅ Python 全景预览柱状图已保存至: {fig_save_path.name}")