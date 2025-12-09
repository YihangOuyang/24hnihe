# 文件名: preprocess_raw_data.py
import os
import pandas as pd

def process_nighttime_data():
    """
    加载原始的光伏数据Excel文件，保留所有时间点，
    但将功率低于阈值的“夜间”数据点的值设置为0，
    然后将处理后的完整DataFrame保存到新的CSV文件中。
    """
    print("--- 开始运行数据预处理脚本：将夜间功率置零 ---")

    # =================================================================
    #                       用户配置区域
    # =================================================================
    
    # 1. 定义原始数据文件的路径
    input_file_path = r'2019_pv_raw.csv'

    # 2. 定义功率列的名称
    power_column_name = 'Power (kW)'

    # 3. 设定功率阈值
    #    任何小于此值的数据点都将被视为“夜间数据”并被置为0。
    power_threshold = 0 # (单位：千瓦特)
    
    # =================================================================
    
    # 定义输出文件的路径和名称
    output_directory = r'putputs/clean'
    output_filename = '2019_pv_daytime_only.csv'
    
    # --- 1. 加载Excel文件 (这部分不变) ---
    try:
        print(f"正在加载原始数据文件: {input_file_path}")
        df_raw = pd.read_csv(input_file_path)
        print(f"加载成功！原始数据共有 {len(df_raw)} 行。")
    except FileNotFoundError:
        print(f"错误：输入文件未找到 '{input_file_path}'。请检查路径。")
        return
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    # --- 2. 检查功率列是否存在 (这部分不变) ---
    if power_column_name not in df_raw.columns:
        print(f"错误：在文件中找不到名为 '{power_column_name}' 的列。")
        print(f"文件中可用的列为: {list(df_raw.columns)}")
        return

    # =================================================================
    # [核心修改] 将低于阈值的功率值设置为0，而不是删除行
    # =================================================================
    print(f"正在根据阈值 ({power_threshold} kW) 将夜间功率置为0...")
    
    # 创建一个布尔条件
    condition = df_raw[power_column_name] < power_threshold
    
    # 使用 .loc 索引器进行条件赋值
    # df_raw.loc[行条件, 列名] = 新值
    df_raw.loc[condition, power_column_name] = 0
    
    # 统计有多少个值被修改了
    values_set_to_zero = condition.sum()
    print(f"处理完成！共有 {values_set_to_zero} 个数据点被置为0。")
    print(f"数据总行数保持不变: {len(df_raw)} 行。")

    # --- 4. 保存处理后的文件 ---
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    full_output_path = os.path.join(output_directory, output_filename)
    
    df_raw.to_csv(full_output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n预处理完成！夜间功率置零后的文件已保存到: {full_output_path}")
    print("--- 数据预处理脚本运行结束 ---")

if __name__ == '__main__':
    process_nighttime_data()