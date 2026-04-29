import pandas as pd
import matplotlib.pyplot as plt
import io

# 你的原始数据
csv_data = """Original_Baseline,None,1.0,4.77
Vanilla_SVD,3,0.6,4.84
Vanilla_SVD,3-4,0.6,5.04
Vanilla_SVD,3-5,0.6,5.73
Vanilla_SVD,3-6,0.6,7.54
Vanilla_SVD,3-7,0.6,8.43
Vanilla_SVD,3-8,0.6,9.94
Vanilla_SVD,3-9,0.6,11.02
Vanilla_SVD,3-10,0.6,12.55
Vanilla_SVD,3-11,0.6,13.01
Vanilla_SVD,3-12,0.6,15.17
Vanilla_SVD,3-13,0.6,18.11
Vanilla_SVD,3-14,0.6,22.0
Vanilla_SVD,3-15,0.6,30.97
Vanilla_SVD,3-16,0.6,43.82
Vanilla_SVD,3-17,0.6,56.07
Vanilla_SVD,3-18,0.6,69.83
Vanilla_SVD,3-19,0.6,71.99
Vanilla_SVD,3-20,0.6,86.43
Vanilla_SVD,3-21,0.6,98.04
Vanilla_SVD,3-22,0.6,109.5
Vanilla_SVD,3-23,0.6,109.49
Vanilla_SVD,3-24,0.6,123.15
Vanilla_SVD,3-25,0.6,131.4
Vanilla_SVD,3-26,0.6,141.31
Vanilla_SVD,3-27,0.6,158.92
Vanilla_SVD,3-28,0.6,161.61
Vanilla_SVD,3-29,0.6,184.21
Vanilla_SVD,3-30,0.6,598.01
Vanilla_SVD,3-31,0.6,598.41
Pure_MPO_NoLoRA,3,0.6,5.01
Pure_MPO_NoLoRA,3-4,0.6,6.24
Pure_MPO_NoLoRA,3-5,0.6,16.29
Pure_MPO_NoLoRA,3-6,0.6,28.45
Pure_MPO_NoLoRA,3-7,0.6,39.53
Pure_MPO_NoLoRA,3-8,0.6,49.2
Pure_MPO_NoLoRA,3-9,0.6,80.2
Pure_MPO_NoLoRA,3-10,0.6,168.54
Pure_MPO_NoLoRA,3-11,0.6,258.85
Pure_MPO_NoLoRA,3-12,0.6,415.96
Pure_MPO_NoLoRA,3-13,0.6,576.72
Pure_MPO_NoLoRA,3-14,0.6,761.09
Pure_MPO_NoLoRA,3-15,0.6,1030.82
Pure_MPO_NoLoRA,3-16,0.6,1152.27
Pure_MPO_NoLoRA,3-17,0.6,1636.35
Pure_MPO_NoLoRA,3-18,0.6,1456.69
Pure_MPO_NoLoRA,3-19,0.6,1486.69
Pure_MPO_NoLoRA,3-20,0.6,1727.0
Pure_MPO_NoLoRA,3-21,0.6,1802.71
Pure_MPO_NoLoRA,3-22,0.6,1969.56
Pure_MPO_NoLoRA,3-23,0.6,1948.83
Pure_MPO_NoLoRA,3-24,0.6,2202.68
Pure_MPO_NoLoRA,3-25,0.6,2180.73
Pure_MPO_NoLoRA,3-26,0.6,2306.12
Pure_MPO_NoLoRA,3-27,0.6,2749.87
Pure_MPO_NoLoRA,3-28,0.6,2549.27
Pure_MPO_NoLoRA,3-29,0.6,2415.81
Pure_MPO_NoLoRA,3-30,0.6,2758.87
Pure_MPO_NoLoRA,3-31,0.6,5641.97
MPO_LoRA_Base,3,0.6,5.01
MPO_LoRA_Base,3-4,0.6,5.58
MPO_LoRA_Base,3-5,0.6,8.09
MPO_LoRA_Base,3-6,0.6,17.51
MPO_LoRA_Base,3-7,0.6,25.06
MPO_LoRA_Base,3-8,0.6,28.2
MPO_LoRA_Base,3-9,0.6,35.09
MPO_LoRA_Base,3-10,0.6,48.7
MPO_LoRA_Base,3-11,0.6,62.03
MPO_LoRA_Base,3-12,0.6,75.47
MPO_LoRA_Base,3-13,0.6,109.94
MPO_LoRA_Base,3-14,0.6,166.52
MPO_LoRA_Base,3-15,0.6,250.08
MPO_LoRA_Base,3-16,0.6,308.99
MPO_LoRA_Base,3-17,0.6,424.89
MPO_LoRA_Base,3-18,0.6,516.11
MPO_LoRA_Base,3-19,0.6,564.69
MPO_LoRA_Base,3-20,0.6,645.13
MPO_LoRA_Base,3-21,0.6,659.0
MPO_LoRA_Base,3-22,0.6,671.29
MPO_LoRA_Base,3-23,0.6,647.01
MPO_LoRA_Base,3-24,0.6,644.87
MPO_LoRA_Base,3-25,0.6,638.15
MPO_LoRA_Base,3-26,0.6,618.42
MPO_LoRA_Base,3-27,0.6,675.33
MPO_LoRA_Base,3-28,0.6,676.12
MPO_LoRA_Base,3-29,0.6,700.78
MPO_LoRA_Base,3-30,0.6,673.12
MPO_LoRA_Base,3-31,0.6,768.38
MPO_LoRA_Permuted,3,0.6,4.98
MPO_LoRA_Permuted,3-4,0.6,5.46
MPO_LoRA_Permuted,3-5,0.6,6.95
MPO_LoRA_Permuted,3-6,0.6,10.66
MPO_LoRA_Permuted,3-7,0.6,17.03
MPO_LoRA_Permuted,3-8,0.6,19.05
MPO_LoRA_Permuted,3-9,0.6,22.47
MPO_LoRA_Permuted,3-10,0.6,28.33
MPO_LoRA_Permuted,3-11,0.6,32.81
MPO_LoRA_Permuted,3-12,0.6,41.23
MPO_LoRA_Permuted,3-13,0.6,56.17
MPO_LoRA_Permuted,3-14,0.6,78.78
MPO_LoRA_Permuted,3-15,0.6,112.74
MPO_LoRA_Permuted,3-16,0.6,144.41
MPO_LoRA_Permuted,3-17,0.6,172.85
MPO_LoRA_Permuted,3-18,0.6,208.98
MPO_LoRA_Permuted,3-19,0.6,244.19
MPO_LoRA_Permuted,3-20,0.6,261.69
MPO_LoRA_Permuted,3-21,0.6,287.27
MPO_LoRA_Permuted,3-22,0.6,305.31
MPO_LoRA_Permuted,3-23,0.6,301.82
MPO_LoRA_Permuted,3-24,0.6,298.04
MPO_LoRA_Permuted,3-25,0.6,298.85
MPO_LoRA_Permuted,3-26,0.6,305.71
MPO_LoRA_Permuted,3-27,0.6,335.59
MPO_LoRA_Permuted,3-28,0.6,341.25
MPO_LoRA_Permuted,3-29,0.6,335.13
MPO_LoRA_Permuted,3-30,0.6,366.29
MPO_LoRA_Permuted,3-31,0.6,378.16"""

# 读取数据
df = pd.read_csv(io.StringIO(csv_data), header=None, names=['Method', 'Layers', 'Ratio', 'Value'])

# 提取 Baseline 数值
baseline_val = df[df['Method'] == 'Original_Baseline']['Value'].values[0]

# 过滤掉 Baseline，处理其他方法的数据
df_methods = df[df['Method'] != 'Original_Baseline'].copy()

# 将 '3-X' 格式转换为 X，作为横坐标
def extract_x(layer_str):
    if '-' in str(layer_str):
        return int(str(layer_str).split('-')[1])
    return int(layer_str)

df_methods['X'] = df_methods['Layers'].apply(extract_x)

# 定义画图的颜色和标记样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
markers = ['o', 's', '^', 'D']

# 开始画图
plt.figure(figsize=(12, 7))

for i, method in enumerate(df_methods['Method'].unique()):
    subset = df_methods[df_methods['Method'] == method]
    plt.plot(subset['X'], subset['Value'], marker=markers[i], color=colors[i], label=method, linewidth=2, markersize=6)

# 画 Baseline (虚线)
plt.axhline(y=baseline_val, color='red', linestyle='--', linewidth=2, label=f'Original_Baseline ({baseline_val})')

# 图表美化与设置
plt.yscale('log') # 使用对数坐标轴，因为数据跨度太大
plt.xlabel('Modified Layers (End Layer)', fontsize=12, fontweight='bold')
plt.ylabel('Value (Log Scale)', fontsize=12, fontweight='bold')
plt.title('Performance Comparison Across Different Methods and Layers', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)

# 调整 X 轴刻度，让它显示 3 到 31
plt.xticks(range(3, 32, 2))

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)