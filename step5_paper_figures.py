# -*- coding: utf-8 -*-
"""
Step 5 (Origin Export): Export Comprehensive Model Comparison Data to CSV
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ================= 配置路径 =================
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_PATH = CURRENT_DIR / "outputs" / "benchmark" / "benchmark_all_predictions_timeseries.csv"
OUT_DIR = CURRENT_DIR / "outputs" / "origin_data"  # 新建一个专属的 Origin 数据文件夹
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(">>> 正在读取 2018-2019年预测结果数据...")
df = pd.read_csv(DATA_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# 过滤掉夜间零值，只保留有效发电时段进行评估
df_day = df[df['Actual_Power_Fixed'] > 0.1].copy()

# 提取模型名称列表
model_cols = [c for c in df.columns if c.startswith('Pred_')]
models = [c.replace('Pred_', '') for c in model_cols]

# 计算所有模型的绝对误差
for model in models:
    df_day[f'Err_{model}'] = np.abs(df_day[f'Pred_{model}'] - df_day['Actual_Power_Fixed'])

# =====================================================================
# 1. 自动天气打标 (Weather Regime Classification)
# =====================================================================
print(">>> 正在进行天气特征聚类打标...")
df_day['Date'] = df_day.index.date
daily_stats = df_day.groupby('Date')['Actual_Power_Fixed'].agg(
    mean_power='mean',
    volatility=lambda x: np.std(np.diff(x)) if len(x) > 1 else 0
)

vol_threshold = daily_stats['volatility'].quantile(0.66)
mean_threshold = daily_stats[daily_stats['volatility'] <= vol_threshold]['mean_power'].median()

def classify_regime(row):
    # 去掉了原先用于 Python 画图的换行符 \n，让 Origin 的图例和坐标轴更干净
    if row['volatility'] > vol_threshold:
        return 'Partly Cloudy' 
    elif row['mean_power'] > mean_threshold:
        return 'Clear Sky'
    else:
        return 'Overcast'

daily_stats['Regime'] = daily_stats.apply(classify_regime, axis=1)
df_day = df_day.merge(daily_stats[['Regime']], left_on='Date', right_index=True)

# =====================================================================
# 导出 1：全模型天气分型箱型图数据 (Grouped Boxplot Data)
# =====================================================================
print(">>> 正在导出 Origin 箱型图专用数据...")
err_cols = [f'Err_{m}' for m in models]

# 使用 melt 转为长表，Origin 处理分组箱线图的最佳格式
df_melt = df_day.melt(id_vars=['Regime'], value_vars=err_cols, 
                      var_name='Model', value_name='Absolute Error (kW)')

# 【核心修改点 1】：正确整理 8 个模型名字，区分基线和你的模型
df_melt['Model'] = df_melt['Model'].str.replace('Err_', '')
df_melt['Model'] = df_melt['Model'].replace({
    'Transformer': 'Transformer(Baseline)', 
    'PiT-Net': 'PiT-Net(Proposed)'
})

# 排序，让数据在 Origin 的 Worksheet 里按天气和模型整齐排列
df_melt.sort_values(by=['Regime', 'Model'], inplace=True)
df_melt.to_csv(OUT_DIR / "origin_boxplot_data.csv", index=False)

# =====================================================================
# 导出 2：y=x 散点对比图数据 (按天气类型分别保存 CSV)
# =====================================================================
print(">>> 正在导出 Origin 散点图专用数据 (按天气分文件)...")

# 提取散点图需要的列：真实功率、各个模型的预测功率
scatter_cols = ['Actual_Power_Fixed'] + model_cols
df_scatter_base = df_day[['Regime'] + scatter_cols].copy()

# 【核心修改点 2】：正确重命名列名，防止在 Origin 里弄混
df_scatter_base.rename(columns={
    'Pred_Transformer': 'Pred_Transformer(Baseline)',
    'Pred_PiT-Net': 'Pred_PiT-Net(Proposed)'
}, inplace=True)

# 获取所有的天气分类 ['Clear Sky', 'Overcast', 'Partly Cloudy']
unique_regimes = df_scatter_base['Regime'].unique()

for regime in unique_regimes:
    # 1. 筛选出当前天气的数据
    df_regime = df_scatter_base[df_scatter_base['Regime'] == regime].copy()
    
    # 2. 按真实功率排序，让 X 轴数据从小到大排列，在 Origin 中管理更清晰
    df_regime.sort_values(by='Actual_Power_Fixed', inplace=True)
    
    # 3. 剔除 Regime 列（因为文件名已经说明了天气），让数据表只包含纯粹的 XY 数据
    df_regime.drop(columns=['Regime'], inplace=True)
    
    # 4. 生成安全的文件名（将空格替换为下划线）
    safe_regime_name = regime.replace(' ', '_')
    save_path = OUT_DIR / f"origin_scatter_yx_{safe_regime_name}.csv"
    
    # 5. 导出该天气的专属 CSV
    df_regime.to_csv(save_path, index=False)
    print(f"   -> 已保存: {save_path.name} (包含 {len(df_regime)} 个散点)")

print(f"\n✅ 数据导出成功！请在 {OUT_DIR} 文件夹中查看。")