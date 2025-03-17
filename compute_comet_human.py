import pandas as pd
import numpy as np
import os
import time
import torch
# 利用Tensor Cores提高性能
torch.set_float32_matmul_precision('medium')
from comet import download_model, load_from_checkpoint

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    print("人类翻译COMET评分比较开始...")
    start_time = time.time()
    
    # 测试模式 - 只处理前100个任务
    TEST_MODE = False
    MAX_TEST_TASKS = 100
    
    # 读取Excel
    df = pd.read_excel("民法典日语翻译（大模型+人类）.xlsx")
    
    # 定义人类翻译列
    human_translators = ["渠涛-日文翻译", "立命馆-日文翻译", "白出-日文翻译"]
    
    # 创建结果DataFrame
    result_df = df.copy()
    
    # 下载并加载Comet模型
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    print(f"已加载Comet模型: {model_path}")
    
    # 准备数据 - 每个人类翻译与其他人类翻译进行比较
    all_data = []
    task_count = 0
    
    for index, row in df.iterrows():
        source_text = row["中文"]  # 源语言文本
        
        # 每位译者作为候选翻译
        for human_mt in human_translators:
            # 其他译者作为参考翻译
            for human_ref in human_translators:
                # 跳过自己与自己的比较
                if human_mt == human_ref:
                    continue
                    
                if pd.isna(row[human_mt]) or pd.isna(row[human_ref]) or pd.isna(source_text):
                    continue
                
                all_data.append({
                    "index": index,
                    "human_mt": human_mt,
                    "human_ref": human_ref,
                    "src": source_text,
                    "mt": row[human_mt],
                    "ref": row[human_ref]
                })
                
                task_count += 1
                
                if TEST_MODE and task_count >= MAX_TEST_TASKS:
                    break
            
            if TEST_MODE and task_count >= MAX_TEST_TASKS:
                break
        
        if TEST_MODE and task_count >= MAX_TEST_TASKS:
            break
    
    print(f"创建了 {len(all_data)} 个评估任务")
    
    # 批量处理，每批次处理64个样本
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
            human_mt = item["human_mt"]
            human_ref = item["human_ref"]
            
            col_name = f"{human_mt}_vs_{human_ref}_HumanCometScore"
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
            temp_file = f"human_comet_results_temp_{processed_count}.xlsx"
            result_df.to_excel(temp_file, index=False)
            print(f"\n已保存临时结果: {temp_file}")
    
    # 计算每个人类翻译者的平均分
    for human_mt in human_translators:
        # 计算当前翻译者与其他翻译者相比的平均分
        other_refs = [ref for ref in human_translators if ref != human_mt]
        comet_cols = [f"{human_mt}_vs_{ref}_HumanCometScore" for ref in other_refs]
        result_df[f"{human_mt}_Avg_HumanCometScore"] = result_df[comet_cols].mean(axis=1)
    
    # 计算整体人类翻译评分矩阵
    human_matrix = np.zeros((len(human_translators), len(human_translators)))
    for i, mt in enumerate(human_translators):
        for j, ref in enumerate(human_translators):
            if mt != ref:
                col_name = f"{mt}_vs_{ref}_HumanCometScore"
                human_matrix[i][j] = result_df[col_name].mean()
    
    # 保存最终结果
    output_file = "民法典日语翻译_人类翻译对比_CometScore.xlsx"
    result_df.to_excel(output_file, index=False)
    
    # 打印统计
    total_time = time.time() - start_time
    print(f"\n\n计算完成! 总用时: {total_time:.1f}秒, 平均每任务: {total_time/len(all_data):.3f}秒")
    
    # 打印人类翻译者之间的评分矩阵
    print("\n人类翻译者之间的COMET评分矩阵:")
    print("行=候选翻译，列=参考翻译")
    print("=" * 60)
    header = "           | " + " | ".join(f"{name[:6]}..." for name in human_translators)
    print(header)
    print("-" * len(header))
    
    for i, mt in enumerate(human_translators):
        row = f"{mt[:6]}... | "
        for j, ref in enumerate(human_translators):
            if mt == ref:
                row += "  --   | "
            else:
                row += f"{human_matrix[i][j]:.4f} | "
        print(row)
    
    print("=" * 60)
    
    # 计算每个人类翻译者的平均得分（作为候选翻译时）
    print("\n各人类翻译者平均COMET得分:")
    for i, mt in enumerate(human_translators):
        # 排除对角线元素（自己与自己比较）
        scores = [human_matrix[i][j] for j in range(len(human_translators)) if i != j]
        avg_score = np.mean(scores)
        print(f"{mt}: {avg_score:.4f}")

if __name__ == "__main__":
    main()