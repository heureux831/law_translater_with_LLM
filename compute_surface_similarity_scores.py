import pandas as pd
from sacrebleu import sentence_bleu
from rouge import Rouge
from jiwer import wer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# 读取Excel文件
df = pd.read_excel("民法典日语翻译（大模型+人类）.xlsx")

# 定义参考列
reference_columns = ["渠涛-日文翻译", "立命馆-日文翻译", "白出-日文翻译"]

# 定义需要评分的列
model_columns = [
    "Kimi-32k日文翻译", "Doubao-32k pro 日文翻译", "Qwen2.5:14b-日文翻译",
    "GPT-4o-日文翻译", "GPT-o1-日文翻译", "DeepSeek-v3-日文翻译", "DeepSeek-R1-日文翻译"
]

# 初始化结果存储
results = []

def calculate_scores(candidate, references):
    """统一计算BLEU、ROUGE和TER分数"""
    # BLEU计算
    bleu_score = round(sentence_bleu(candidate, references, tokenize="char").score, 2)
    
    # ROUGE计算
    rouge = Rouge()
    rouge_scores = []
    for ref in references:
        scores = rouge.get_scores(
            ' '.join(list(candidate)), 
            ' '.join(list(ref))
        )
        rouge_scores.append(round(scores[0]['rouge-l']['f']*100, 2))
    avg_rouge = round(sum(rouge_scores) / len(rouge_scores) , 2)
    
    # TER计算
    ter_scores = []
    for ref in references:
        ref_chars = ' '.join(list(ref))
        can_chars = ' '.join(list(candidate))
        ter_scores.append(round(wer(ref_chars, can_chars)*100, 2))
    avg_ter = round(sum(ter_scores) / len(ter_scores), 2)
    
    return bleu_score, avg_rouge, avg_ter

def process_single_task(args):
    """单个计算任务的并行处理"""
    index, row_dict, model_col = args
    references = [row_dict[col] for col in reference_columns]
    candidate = row_dict[model_col]
    
    bleu, rouge, ter = calculate_scores(candidate, references)
    
    return (index + 1, model_col, bleu, rouge, ter)

# 主执行逻辑
if __name__ == '__main__':
    # 准备所有计算任务（行号，行数据字典，模型列）
    tasks = [
        (idx, row.to_dict(), model_col)
        for idx, row in df.iterrows()
        for model_col in model_columns
    ]
    
    # 使用多进程池并行处理
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}
        
        results = []
        progress = tqdm(total=len(tasks), desc="并行计算进度")
        
        for future in as_completed(futures):
            try:
                row_num, model, bleu, rouge, ter = future.result()
                results.append({
                    "行号": row_num,
                    "模型": model,
                    "BLEU": round(bleu, 2),
                    "ROUGE-L F1": round(rouge, 2),
                    "TER": round(ter, 2)
                })
                progress.update(1)
            except Exception as e:
                print(f"任务出错: {str(e)}")
        progress.close()

    # 将结果保存为新的Excel文件
    result_df = pd.DataFrame(results)
    result_df.to_excel("翻译评分结果.xlsx", index=False)

    print("评分计算完成，结果已保存到 '翻译评分结果.xlsx'")

# 读取原始评分结果
df = pd.read_excel("翻译评分结果.xlsx")

# 重新组织数据
# 首先按行号分组
pivoted_data = {}
metrics = ['BLEU', 'ROUGE-L F1', 'TER']

for row_num in df['行号'].unique():
    row_data = df[df['行号'] == row_num]
    row_dict = {'行号': row_num}
    
    # 为每个模型添加所有指标
    for _, model_row in row_data.iterrows():
        model_name = model_row['模型']
        for metric in metrics:
            col_name = f"{model_name}_{metric}"
            row_dict[col_name] = model_row[metric]
    
    pivoted_data[row_num] = row_dict

# 创建新的DataFrame
new_df = pd.DataFrame.from_dict(pivoted_data, orient='index')

# 保存重组后的结果
new_df.to_excel("重组后的翻译评分结果.xlsx", index=False) 