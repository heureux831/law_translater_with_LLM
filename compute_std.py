import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 读取重组后的评分结果
df = pd.read_excel("重组后的翻译评分结果.xlsx")

# 获取所有模型名称（修复可能漏掉模型的情况）
model_names = list(set([col.split('_')[0] for col in df.columns if '_' in col]))
print("检测到的模型列表:", model_names)  # 新增调试信息
metrics = ['BLEU', 'ROUGE-L F1', 'TER']

# 创建结果存储字典
stats_results = {}

for model in model_names:
    model_stats = {}
    for metric in metrics:
        col_name = f"{model}_{metric}"
        values = df[col_name]
        
        # 基础统计量
        model_stats[f"{metric}_均值"] = np.mean(values)
        model_stats[f"{metric}_方差"] = np.var(values)
        model_stats[f"{metric}_标准差"] = np.std(values)
        model_stats[f"{metric}_中位数"] = np.median(values)
        
        # 分位数
        q1, q3 = np.percentile(values, [25, 75])
        model_stats[f"{metric}_Q1"] = q1
        model_stats[f"{metric}_Q3"] = q3
        model_stats[f"{metric}_IQR"] = q3 - q1
        
        # 偏度和峰度
        model_stats[f"{metric}_偏度"] = stats.skew(values)
        model_stats[f"{metric}_峰度"] = stats.kurtosis(values)
        
        # 置信区间
        ci = stats.t.interval(confidence=0.95, 
                            df=len(values)-1,
                            loc=np.mean(values),
                            scale=stats.sem(values))
        model_stats[f"{metric}_95%置信区间下限"] = ci[0]
        model_stats[f"{metric}_95%置信区间上限"] = ci[1]
        
        # 变异系数（CV）
        model_stats[f"{metric}_变异系数"] = stats.variation(values)
        
    stats_results[model] = model_stats

# 转换为DataFrame并保存
stats_df = pd.DataFrame.from_dict(stats_results, orient='index')
stats_df.to_excel("翻译评分统计分析.xlsx")

# 打印每个模型的关键指标
for model in model_names:
    print(f"\n{model} 的主要统计指标：")
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"均值: {stats_results[model][f'{metric}_均值']:.4f}")
        print(f"标准差: {stats_results[model][f'{metric}_标准差']:.4f}")
        print(f"95%置信区间: [{stats_results[model][f'{metric}_95%置信区间下限']:.4f}, "
              f"{stats_results[model][f'{metric}_95%置信区间上限']:.4f}]")
        print(f"变异系数: {stats_results[model][f'{metric}_变异系数']:.4f}")

# 可视化设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Songti SC']  # 多备选字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
colors = sns.color_palette("husl", n_colors=len(stats_df))

# 在可视化设置部分添加：
system_fonts = matplotlib.font_manager.findSystemFonts()
chinese_fonts = [f for f in system_fonts if any(name in f.lower() for name in ['hei', 'yahei', 'songti'])]
if chinese_fonts:
    plt.rcParams['font.family'] = matplotlib.font_manager.FontProperties(fname=chinese_fonts[0]).get_name()

# 1. 综合对比雷达图
metrics = ['BLEU_均值', 'ROUGE-L F1_均值', 'TER_均值']
labels = ['BLEU', 'ROUGE-L', 'TER']

# 收集所有指标值用于归一化
all_values = stats_df[metrics].values
# 对TER进行反向处理（因为TER越小越好）
all_values[:, 2] = 1 / (all_values[:, 2] + 1e-6)  # 更稳定的反向处理

# 计算全局最大最小值
global_min = all_values.min(axis=0)
global_max = all_values.max(axis=0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

for idx, (model, row) in enumerate(stats_df.iterrows()):
    values = row[metrics].values.copy()
    # 处理TER指标（取倒数并统一归一化）
    values[2] = 1 / (values[2] + 1e-6)  # 避免除零
    # 使用全局范围进行归一化
    normalized = (values - global_min) / (global_max - global_min)
    
    # 闭合曲线（添加第一个点的副本到末尾）
    angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)
    values_plot = np.concatenate((normalized, [normalized[0]]))
    angles_plot = np.concatenate((angles, [angles[0]]))
    
    ax.plot(angles_plot, values_plot, 
            color=colors[idx], linewidth=2, label=model)
    ax.fill(angles_plot, values_plot, 
           color=colors[idx], alpha=0.1)

# 设置标签位置（添加闭合标签）
ax.set_xticks(np.concatenate((angles, [angles[0]])))
ax.set_xticklabels(labels + [labels[0]])
ax.set_title("模型性能雷达图（归一化处理）")
plt.legend(bbox_to_anchor=(1.2, 1))
plt.savefig('radar_chart.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. 置信区间对比图
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
for i, metric in enumerate(['BLEU', 'ROUGE-L F1', 'TER']):
    # 准备数据
    means = stats_df[f'{metric}_均值']
    lowers = stats_df[f'{metric}_95%置信区间下限']
    uppers = stats_df[f'{metric}_95%置信区间上限']
    
    # 排序
    sorted_idx = means.argsort()
    if metric == 'TER':  # TER越小越好，倒序排序
        sorted_idx = sorted_idx[::-1]
    
    # 绘制
    axes[i].errorbar(means[sorted_idx], means.index[sorted_idx], 
                    xerr=[means[sorted_idx]-lowers[sorted_idx], 
                         uppers[sorted_idx]-means[sorted_idx]],
                    fmt='o', color='darkred', ecolor='lightcoral', 
                    elinewidth=3, capsize=0)
    axes[i].set_title(f'{metric}指标对比（95%置信区间）')
    axes[i].set_xlabel('Score' if metric != 'TER' else 'Error Rate')
plt.tight_layout()
plt.savefig('confidence_intervals.png', dpi=300)
plt.close()

# 3. 统计显著性检验（以BLEU为例）
print("\nBLEU分数T检验结果矩阵：")
p_value_matrix = pd.DataFrame(index=stats_df.index, columns=stats_df.index)
for model1 in stats_df.index:
    for model2 in stats_df.index:
        if model1 != model2:
            # 获取两组数据的均值和样本量（假设样本量相同）
            n = len(df)  # 需要从原始数据获取实际样本量
            t_stat, p_val = stats.ttest_ind_from_stats(
                stats_df.loc[model1, 'BLEU_均值'], stats_df.loc[model1, 'BLEU_标准差'], n,
                stats_df.loc[model2, 'BLEU_均值'], stats_df.loc[model2, 'BLEU_标准差'], n
            )
            p_value_matrix.loc[model1, model2] = f"{p_val:.4f}"
print(p_value_matrix)
