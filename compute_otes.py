import pandas as pd
import os
import numpy as np
import json
import time
import asyncio
from dotenv import load_dotenv
import threading
import httpx
from openai import AsyncOpenAI

# 加载环境变量
load_dotenv()

# 设置API信息
api_key = os.getenv("OPENAI_API_KEY", "sk-s6eKyW03tJ3urYixhq1NpQknFxKc1frRnBXJCDzPTdO4cYqi")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.nekoapi.com/v1")

# 测试模式 - 只处理前100个任务
TEST_MODE = False
MAX_TEST_TASKS = 100

# 线程安全的变量
cache_lock = threading.Lock()

# 定义嵌入模型
EMBEDDING_MODEL = "text-embedding-3-small"

# 加载缓存
embedding_cache = {}
cache_file = "embedding_cache.json"
if os.path.exists(cache_file):
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            embedding_cache = json.load(f)
        print(f"已加载 {len(embedding_cache)} 条嵌入缓存")
    except:
        print("缓存文件加载失败，将创建新缓存")

# 计算余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 任务类
class EmbeddingTask:
    def __init__(self, index, model_col, ref_col, candidate_text, reference_text):
        self.index = index
        self.model_col = model_col
        self.ref_col = ref_col
        self.candidate_text = candidate_text
        self.reference_text = reference_text
        self.similarity = None

# 异步获取嵌入向量
async def get_embedding_async(client, text, model=EMBEDDING_MODEL):
    cache_key = f"{model}:{text}"
    
    # 检查缓存
    with cache_lock:
        if cache_key in embedding_cache:
            return embedding_cache[cache_key]
    
    try:
        response = await client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )
        embedding = response.data[0].embedding
        
        # 更新缓存
        with cache_lock:
            embedding_cache[cache_key] = embedding
            
            # 每10个新项目保存一次（测试模式更频繁保存）
            if len(embedding_cache) % 500 == 0:
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(embedding_cache, f, ensure_ascii=False)
                except Exception as e:
                    print(f"保存缓存时出错: {e}")
        
        return embedding
    except Exception as e:
        print(f"获取嵌入向量出错: {str(e)[:100]}...")
        await asyncio.sleep(1)  # 错误时暂停
        return None

# 异步处理单个任务
async def process_task_async(client, task, semaphore):
    async with semaphore:  # 使用信号量控制并发
        try:
            # 异步获取两个嵌入向量
            candidate_embedding = await get_embedding_async(client, task.candidate_text)
            reference_embedding = await get_embedding_async(client, task.reference_text)
            
            if candidate_embedding and reference_embedding:
                task.similarity = cosine_similarity(candidate_embedding, reference_embedding)
            else:
                task.similarity = np.nan
        except Exception as e:
            print(f"处理任务出错: {e}")
            task.similarity = np.nan
        
        return task

# 主异步函数
async def main_async():
    print("OpenAI AsyncAPI处理开始...")
    print(f"测试模式: {'启用，仅处理前 '+str(MAX_TEST_TASKS)+' 个任务' if TEST_MODE else '禁用'}")
    start_time = time.time()
    
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
    
    # 一次性创建所有任务
    all_tasks = []
    for index, row in df.iterrows():
        for model_col in model_columns:
            for ref_col in reference_columns:
                if pd.isna(row[model_col]) or pd.isna(row[ref_col]):
                    continue
                
                task = EmbeddingTask(
                    index=index,
                    model_col=model_col,
                    ref_col=ref_col,
                    candidate_text=row[model_col],
                    reference_text=row[ref_col]
                )
                all_tasks.append(task)
                
                # 测试模式下，只收集有限数量的任务
                if TEST_MODE and len(all_tasks) >= MAX_TEST_TASKS:
                    break
            
            if TEST_MODE and len(all_tasks) >= MAX_TEST_TASKS:
                break
        
        if TEST_MODE and len(all_tasks) >= MAX_TEST_TASKS:
            break
    
    # 如果是测试模式，仅保留前MAX_TEST_TASKS个任务
    if TEST_MODE:
        all_tasks = all_tasks[:MAX_TEST_TASKS]
    
    total_tasks = len(all_tasks)
    print(f"创建了 {total_tasks} 个任务")
    
    # 创建异步HTTP客户端 - 使用httpx代替aiohttp
    transport = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=100))
    
    # 降低测试模式下的超时时间，便于快速发现问题
    timeout = 40.0 if TEST_MODE else 30.0
    
    async with httpx.AsyncClient(transport=transport, timeout=timeout) as http_client:
        # 创建AsyncOpenAI客户端，使用httpx.AsyncClient
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=3
        )        
        # 设置最大并发数（测试模式使用更小的并发数）
        max_concurrency = 80 if TEST_MODE else 200
        semaphore = asyncio.Semaphore(max_concurrency)
        print(f"最大并发请求数: {max_concurrency}")
        
        # 创建所有任务的异步协程
        tasks_coroutines = [process_task_async(client, task, semaphore) for task in all_tasks]
        
        # 进度跟踪
        processed_count = 0
        completed_indices = set()  # 跟踪已完成的任务索引
        
        # 用于调试的详细任务信息（仅测试模式）
        task_results = {}
        
        # 处理任务
        for completed_task in asyncio.as_completed(tasks_coroutines):
            # 等待任务完成
            task = await completed_task
            
            # 更新结果
            col_name = f"{task.model_col}_vs_{task.ref_col}_EmbeddingScore"
            if col_name not in result_df.columns:
                result_df[col_name] = np.nan
            
            # 测试模式下，保存所有任务的详细信息
            if TEST_MODE:
                key = (task.index, task.model_col, task.ref_col)
                task_results[key] = {
                    "index": task.index,
                    "model": task.model_col,
                    "reference": task.ref_col,
                    "similarity": task.similarity,
                    "candidate_text": task.candidate_text[:30] + "..." if len(task.candidate_text) > 30 else task.candidate_text,
                    "reference_text": task.reference_text[:30] + "..." if len(task.reference_text) > 30 else task.reference_text
                }
            
            if not pd.isna(task.similarity):  # 确保结果有效
                result_df.at[task.index, col_name] = task.similarity
                # 记录已处理的索引
                key = (task.index, task.model_col, task.ref_col)
                completed_indices.add(key)
            
            # 更新进度
            processed_count += 1
            elapsed = time.time() - start_time
            if processed_count % 100 == 0 or processed_count == total_tasks:  # 测试模式下更频繁地更新
                avg_time = elapsed / processed_count
                eta = avg_time * (total_tasks - processed_count)
                
                # 计算有效结果数量
                valid_results = sum(1 for key in completed_indices if not pd.isna(result_df.at[key[0], f"{key[1]}_vs_{key[2]}_EmbeddingScore"]))
                
                print(f"\r进度: {processed_count}/{total_tasks} ({processed_count/total_tasks*100:.1f}%), "
                      f"有效结果: {valid_results}/{len(completed_indices)}, "
                      f"已用时: {elapsed:.1f}秒, 估计剩余: {eta:.1f}秒", end="", flush=True)
            
            # 每10个任务保存临时结果（测试模式更频繁保存）
            if processed_count % 500 == 0 or processed_count == total_tasks:
                try:
                    # 验证结果是否写入成功
                    result_count = 0
                    for model_col in model_columns:
                        for ref_col in reference_columns:
                            col = f"{model_col}_vs_{ref_col}_EmbeddingScore"
                            if col in result_df.columns:
                                result_count += result_df[col].notna().sum()
                    
                    temp_file = f"embedding_results_test_{processed_count}_{result_count}.xlsx"
                    result_df.to_excel(temp_file, index=False)
                    print(f"\n已保存临时结果: {temp_file}，包含 {result_count} 个有效得分")
                    
                    # 如果发现结果数量明显少于已处理任务数，打印详细信息
                    if result_count < processed_count * 0.9:  # 如果有效结果不到处理数的90%
                        print(f"警告: 有效结果数 ({result_count}) 明显少于已处理任务数 ({processed_count})")
                        # 记录一些样本数据以便调试
                        sample_keys = list(completed_indices)[:5]
                        print(f"样本数据检查 (前5条):")
                        for idx, model, ref in sample_keys:
                            col = f"{model}_vs_{ref}_EmbeddingScore"
                            val = result_df.at[idx, col]
                            print(f"  索引={idx}, 列={col}, 值={'NaN' if pd.isna(val) else val}")
                except Exception as e:
                    print(f"\n保存临时结果失败: {e}")
    
    # 测试模式下，保存详细任务结果用于分析
    if TEST_MODE and task_results:
        try:
            task_info_df = pd.DataFrame(task_results.values())
            task_info_df.to_excel("embedding_task_details.xlsx", index=False)
            print("\n已保存详细任务分析结果至 embedding_task_details.xlsx")
        except Exception as e:
            print(f"\n保存任务详情失败: {e}")
    
    # 保存缓存
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_cache, f, ensure_ascii=False)
        print(f"\n缓存已保存，共 {len(embedding_cache)} 个项目")
    except Exception as e:
        print(f"\n保存缓存失败: {e}")
    
    # 计算每个模型的平均分
    for model_col in model_columns:
        embedding_cols = [f"{model_col}_vs_{ref_col}_EmbeddingScore" for ref_col in reference_columns]
        result_df[f"{model_col}_Avg_EmbeddingScore"] = result_df[embedding_cols].mean(axis=1)
    
    # 保存最终结果
    output_file = "民法典日语翻译_嵌入相似度评分_TEST.xlsx" if TEST_MODE else "民法典日语翻译_嵌入相似度评分_AsyncAPI.xlsx"
    result_df.to_excel(output_file, index=False)
    
    # 打印统计
    total_time = time.time() - start_time
    print(f"\n\n计算完成! 总用时: {total_time:.1f}秒, 平均每任务: {total_time/total_tasks:.3f}秒")
    
    # 打印各模型平均分
    print("\n各模型嵌入相似度平均分：")
    for model_col in model_columns:
        avg_score = result_df[f"{model_col}_Avg_EmbeddingScore"].mean()
        print(f"{model_col}: {avg_score:.4f}")

# 主函数
def main():
    # 在标准Python环境中运行异步代码
    asyncio.run(main_async())

if __name__ == "__main__":
    main()