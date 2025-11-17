import os, subprocess
import warnings

# ------------------ 1. 用户配置 ------------------

# 【重要】确保此路径指向您解压 wgrib2.exe 和所有 .dll 文件的位置
# 如果 .exe 和 .dll 都在此 .py 脚本的同一目录，则使用 r".\wgrib2.exe"
WGRIB2 = r".\wgrib2.exe"

# 那个报告错误的特定文件
BAD_FILE = r"G:\data\GFS\Stanford_2019\gfs.0p25.2019010118.f021.grib2"

# 您的坐标
LAT = 37.4275
LON = -122.1697
LON360 = LON if LON >= 0 else LON + 360.0

# 我们要逐个测试的变量 (wgrib2 -match 正则表达式)
VARIABLES_TO_TEST = {
    "dswrf": ":DSWRF:surface:",
    "dlwrf": ":DLWRF:surface:",
    "t2m":   ":TMP:2 m above ground:",
    "rh2m":  ":RH:2 m above ground:",
    "u10":   ":UGRD:10 m above ground:",
    "v10":   ":VGRD:10 m above ground:",
}

# ------------------ 2. 测试逻辑 ------------------

def run_test():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("="*60)
    print("开始逐个测试变量，以找出导致 'to_we_sn_scan' 崩溃的变量...")
    print(f"测试文件: {BAD_FILE}")
    print("="*60)

    wgrib2_full_path = os.path.abspath(WGRIB2)
    if not os.path.exists(wgrib2_full_path):
        print(f"!!! 致命错误: 找不到 wgrib2.exe")
        print(f"    检查路径: {wgrib2_full_path}")
        return

    cwd = os.path.dirname(wgrib2_full_path)
    exe_name = os.path.basename(WGRIB2)

    for key, match_regex in VARIABLES_TO_TEST.items():
        print(f"\n--- 正在测试: {key} ({match_regex}) ---")
        
        cmd = [
            exe_name, BAD_FILE,
            "-match", match_regex,
            "-lon", f"{LON360}", f"{LAT}",
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, 
                                 encoding='utf-8', errors='ignore',
                                 cwd=cwd, shell=True)
            
            # 检查 wgrib2 是否执行成功
            if res.returncode == 0:
                print(f"  [ ✓ SUCCESS ] {key} 提取成功。")
            else:
                print(f"  [ X FAILED ] {key} 提取失败，代码: {res.returncode}")
                if "to_we_sn_scan" in res.stderr:
                    print(f"  [ !!! 找到元凶 !!! ] 错误信息包含 'to_we_sn_scan'")
                print(f"  Stderr: {res.stderr.strip()}")
                
        except Exception as e:
            print(f"  [ X FAILED ] {key} 发生 Python 异常: {e}")

    print("\n" + "="*60)
    print("测试完成。")

if __name__ == "__main__":
    run_test()