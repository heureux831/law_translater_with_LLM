import pandas as pd
from sacrebleu import sentence_bleu
from rouge import Rouge
from jiwer import wer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import sys
import torch
import subprocess
import importlib
import time

# 检查并安装缺失的依赖
required_packages = ['fugashi', 'ipadic', 'unidic-lite', 'bert-score', 'transformers']
for package in required_packages:
    try:
        importlib.import_module(package.replace('-', '_'))  # 替换包名中的破折号为下划线
    except ImportError:
        print(f"正在安装缺失的依赖: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 安装完成")

# 设置Hugging Face镜像，解决国内访问问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 可选其他镜像：
# os.environ['HF_ENDPOINT'] = 'https://mirror.tuna.tsinghua.edu.cn/huggingface'
# os.environ['HF_ENDPOINT'] = 'https://mirrors.aliyun.com/huggingface'
# os.environ['HF_ENDPOINT'] = 'https://mirrors.ustc.edu.cn/huggingface'

# 可选：设置缓存路径
# os.environ['HF_HOME'] = './huggingface_cache'
# os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache/transformers'

# 导入bert_score (在设置环境变量后导入)
from bert_score import score, BERTScorer

# 读取Excel文件
df = pd.read_excel("民法典日语翻译（大模型+人类）.xlsx")

# 定义参考列
reference_columns = ["渠涛-日文翻译", "立命馆-日文翻译", "白出-日文翻译"]

# 定义需要评分的列
model_columns = [
    "Kimi-32k日文翻译", "Doubao-32k pro 日文翻译", "Qwen2.5:14b-日文翻译",
    "GPT-4o-日文翻译", "GPT-o1-日文翻译", "DeepSeek-v3-日文翻译", "DeepSeek-R1-日文翻译"
]

# 速度优化1: 使用更轻量级的模型
model_name = "tohoku-nlp/bert-base-japanese-v3"  # 使用较小的日语BERT模型

# 速度优化2: 检查并使用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 速度优化3: 预先加载模型一次
print("加载BERT-Score模型...")
scorer = BERTScorer(
    model_type=model_name,
    lang="ja",
    device=device,
    rescale_with_baseline=False,
    num_layers=9  # 使用较少的层
)

# 速度优化4: 批量处理评分
BATCH_SIZE = 64  # 批量大小，可以根据内存调整

def calculate_bertscore_batch(candidates, references):
    """批量计算BertScore"""
    try:
        P, R, F1 = scorer.score(candidates, references)
        return F1.cpu().numpy()
    except Exception as e:
        print(f"计算BertScore时出错: {e}")
        return [np.nan] * len(candidates)

# 计算总任务数量（用于显示整体进度）
total_tasks = 0
valid_tasks = 0
for index, row in df.iterrows():
    for model_col in model_columns:
        for ref_col in reference_columns:
            if not (pd.isna(row[model_col]) or pd.isna(row[ref_col])):
                valid_tasks += 1
            total_tasks += 1

print(f"总计数据: {len(df)}行 × {len(model_columns)}模型 × {len(reference_columns)}参考 = {total_tasks}组")
print(f"有效数据: {valid_tasks}组 (排除空值)")
print("=" * 60)

# 添加时间信息的进度显示
start_time = time.time()
processed_count = 0

# 主处理流程
print("准备数据...")
result_df = df.copy()

# 主处理流程部分的修改
for ref_col_idx, ref_col in enumerate(reference_columns):
    print(f"\n处理参考[{ref_col_idx+1}/{len(reference_columns)}]: {ref_col}")
    
    for i in range(0, len(df), BATCH_SIZE):
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(df))))
        batch_end = min(i + BATCH_SIZE, len(df))
        print(f"  处理批次: {i+1}-{batch_end}/{len(df)}", end="", flush=True)
        
        # 准备批量数据
        ref_batch = [df.at[idx, ref_col] for idx in batch_indices if not pd.isna(df.at[idx, ref_col])]
        if not ref_batch:  # 如果全是空值，跳过
            continue
            
        valid_indices = []
        for model_col in model_columns:
            candidates = []
            valid_batch_indices = []
            
            for j, idx in enumerate(batch_indices):
                if pd.isna(df.at[idx, ref_col]) or pd.isna(df.at[idx, model_col]):
                    continue
                candidates.append(df.at[idx, model_col])
                valid_batch_indices.append(idx)
            
            if not candidates:  # 如果没有有效数据，跳过
                continue
                
            # 复制参考文本以匹配候选文本数量
            references = [df.at[idx, ref_col] for idx in valid_batch_indices]
            
            # 批量计算
            scores = calculate_bertscore_batch(candidates, references)
            
            # 保存结果
            col_name = f"{model_col}_vs_{ref_col}_BertScore"
            if col_name not in result_df.columns:
                result_df[col_name] = np.nan
                
            for idx, score in zip(valid_batch_indices, scores):
                result_df.at[idx, col_name] = score.item() if hasattr(score, 'item') else score

        # 批处理完成后的进度更新
        processed_count += len(valid_batch_indices)
        elapsed = time.time() - start_time
        if len(valid_batch_indices) > 0:
            avg_time = elapsed / processed_count if processed_count > 0 else 0
            eta = avg_time * (valid_tasks - processed_count)
            
            # 更新同一行进度信息
            print(f"\r  处理批次: {i+1}-{batch_end}/{len(df)}, 模型: {ref_col_idx+1}/{len(reference_columns)}, "
                  f"处理: {processed_count}/{valid_tasks}, "
                  f"已用时: {elapsed:.1f}秒, 估计剩余: {eta:.1f}秒", end="", flush=True)
        
        print()  # 每批次结束后换行

# 计算每个模型的平均BertScore（相对于所有参考）
for model_col in model_columns:
    bert_cols = [f"{model_col}_vs_{ref_col}_BertScore" for ref_col in reference_columns]
    result_df[f"{model_col}_Avg_BertScore"] = result_df[bert_cols].mean(axis=1)

# 保存结果
output_file = "民法典日语翻译_BertScore评分_优化.xlsx"
result_df.to_excel(output_file, index=False)
print(f"评分结果已保存至: {output_file}")

# 输出整体平均分数
print("\n各模型BertScore平均分：")
for model_col in model_columns:
    avg_score = result_df[f"{model_col}_Avg_BertScore"].mean()
    print(f"{model_col}: {avg_score:.4f}")

# 结束时的总结信息
total_time = time.time() - start_time
print(f"\n计算完成! 总用时: {total_time:.1f}秒, 平均每条: {total_time/processed_count:.3f}秒")

