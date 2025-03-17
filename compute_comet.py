import pandas as pd
import numpy as np
import os
import time
import torch
# 添加这一行以利用Tensor Cores提高性能
torch.set_float32_matmul_precision('medium')
from comet import download_model, load_from_checkpoint

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    print("Comet 翻译评估开始...")
    start_time = time.time()
    
    # 测试模式 - 只处理前100个任务
    TEST_MODE = False
    MAX_TEST_TASKS = 100
    
    # 读取Excel
    df = pd.read_excel("民法典日语翻译（大模型+人类）.xlsx")
    
    # 定义参考列和模型列
    reference_columns = ["渠涛-日文翻译", "立命馆-日文翻译", "白出-日文翻译"]
    model_columns = [
        "Kimi-32k日文翻译", "Doubao-32k pro 日文翻译", "Qwen2.5:14b-日文翻译",
        "GPT-4o-日文翻译", "GPT-o1-日文翻译", "DeepSeek-v3-日文翻译", "DeepSeek-R1-日文翻译"
    ]
    
    # 创建结果DataFrame
    result_df = df.copy()
    
    # 下载并加载Comet模型
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    print(f"已加载Comet模型: {model_path}")
    
    # 准备数据
    all_data = []
    task_count = 0
    
    for index, row in df.iterrows():
        source_text = row["中文"]  # 源语言文本
        
        for model_col in model_columns:
            for ref_col in reference_columns:
                if pd.isna(row[model_col]) or pd.isna(row[ref_col]) or pd.isna(source_text):
                    continue
                
                all_data.append({
                    "index": index,
                    "model_col": model_col,
                    "ref_col": ref_col,
                    "src": source_text,
                    "mt": row[model_col],
                    "ref": row[ref_col]
                })
                
                task_count += 1
                
                if TEST_MODE and task_count >= MAX_TEST_TASKS:
                    break
            
            if TEST_MODE and task_count >= MAX_TEST_TASKS:
                break
        
        if TEST_MODE and task_count >= MAX_TEST_TASKS:
            break
    
    print(f"创建了 {len(all_data)} 个评估任务")
    
    # 批量处理，每批次处理32个样本
    batch_size = 64
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    
    processed_count = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_data))
        batch_data = all_data[start_idx:end_idx]
        
        # 准备Comet输入格式
        comet_data = []
        for item in batch_data:
            comet_data.append({
                "src": item["src"],
                "mt": item["mt"],
                "ref": item["ref"]
            })
        
        # 使用Comet模型进行评分
        model_output = model.predict(comet_data, batch_size=len(comet_data), gpus=1 if torch.cuda.is_available() else 0)
        
        # 获取system_score (整个批次的平均分)
        batch_score = float(model_output.system_score)
        
        # 输出model_output的所有属性（仅第一个批次）
        if batch_idx == 0:
            print("\nCOMET模型输出属性:")
            for attr in dir(model_output):
                if not attr.startswith('_'):
                    attr_value = getattr(model_output, attr)
                    print(f"- {attr}: {type(attr_value)}")
        
        # 尝试获取单个分数
        try:
            # 首先尝试获取scores属性
            if hasattr(model_output, 'scores'):
                individual_scores = model_output.scores
                print(f"\n使用model_output.scores: {len(individual_scores)}个单独分数")
            # 然后尝试获取seg_scores属性
            elif hasattr(model_output, 'seg_scores'):
                individual_scores = model_output.seg_scores
                print(f"\n使用model_output.seg_scores: {len(individual_scores)}个单独分数")
            else:
                # 如果没有单独分数，则每个样本使用批次平均分
                print(f"\n无法找到单独分数，将为所有{len(batch_data)}个样本使用批次平均分: {batch_score}")
                individual_scores = [batch_score] * len(batch_data)
        except Exception as e:
            print(f"\n获取单独分数时出错: {e}")
            individual_scores = [batch_score] * len(batch_data)
        
        # 将结果写入DataFrame
        for i, item in enumerate(batch_data):
            index = item["index"]
            model_col = item["model_col"]
            ref_col = item["ref_col"]
            
            col_name = f"{model_col}_vs_{ref_col}_CometScore"
            if col_name not in result_df.columns:
                result_df[col_name] = np.nan
            
            # 使用对应的分数或批次平均分
            score = individual_scores[i] if i < len(individual_scores) else batch_score
            result_df.at[index, col_name] = score
        
        # 更新进度
        processed_count += len(batch_data)
        elapsed = time.time() - start_time
        avg_time = elapsed / processed_count
        eta = avg_time * (len(all_data) - processed_count)
        
        print(f"\r进度: {processed_count}/{len(all_data)} ({processed_count/len(all_data)*100:.1f}%), "
              f"已用时: {elapsed:.1f}秒, 估计剩余: {eta:.1f}秒", end="", flush=True)
        
        # 每批次保存临时结果
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            temp_file = f"comet_results_temp_{processed_count}.xlsx"
            result_df.to_excel(temp_file, index=False)
            print(f"\n已保存临时结果: {temp_file}")
    
    # 计算每个模型的平均分
    for model_col in model_columns:
        comet_cols = [f"{model_col}_vs_{ref_col}_CometScore" for ref_col in reference_columns]
        result_df[f"{model_col}_Avg_CometScore"] = result_df[comet_cols].mean(axis=1)
    
    # 保存最终结果
    output_file = "民法典日语翻译_Comet评分_TEST.xlsx" if TEST_MODE else "民法典日语翻译_Comet评分.xlsx"
    result_df.to_excel(output_file, index=False)
    
    # 打印统计
    total_time = time.time() - start_time
    print(f"\n\n计算完成! 总用时: {total_time:.1f}秒, 平均每任务: {total_time/len(all_data):.3f}秒")
    
    # 打印各模型平均分
    print("\n各模型Comet平均分：")
    for model_col in model_columns:
        avg_score = result_df[f"{model_col}_Avg_CometScore"].mean()
        print(f"{model_col}: {avg_score:.4f}")

if __name__ == "__main__":
    main()