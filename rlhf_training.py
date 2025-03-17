import openai
import os
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from transformers import DataCollatorForSeq2Seq

openai.api_key = os.getenv("OPENAI_API_KEY")

def compute_reward(generated_text: str, reference_text: str) -> float:
    """
    使用 OpenAI API 评分翻译质量，返回奖励值（0~1）
    """
    prompt = f"请对以下翻译进行评分，分数范围为 0~1，其中 1 表示完全正确，0 表示完全错误。\n\n\
    生成的翻译: {generated_text}\n参考翻译: {reference_text}\n\n请只输出一个数值作为评分。"
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    try:
        reward = float(response["choices"][0]["message"]["content"].strip())
    except ValueError:
        reward = 0.0
    return reward



# PPO 训练配置
ppo_config = PPOConfig(
    batch_size=16,         # 每批训练样本数
    learning_rate=1.41e-5, # PPO 学习率
    mini_batch_size=4,     # mini batch 训练
    optimize_for_causal_lm=True, # 用于自回归模型
)

# 将原始模型转换为 PPO 可训练的模型
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)
ppo_trainer = PPOTrainer(ppo_config, ppo_model, tokenizer=tokenizer)

# 训练数据
train_data = [
    {"input": "The patient has a heart attack.", "output": "患者出现了心肌梗死。"},
    {"input": "He suffered from a stroke.", "output": "他患了中风。"}
]

# PPO 训练循环
for epoch in range(ppo_config.total_episodes):
    batch = train_data[:ppo_config.batch_size]  # 取一个 batch
    queries = [sample["input"] for sample in batch]  # 输入文本
    responses = [ppo_model.generate(tokenizer.encode(q, return_tensors="pt")) for q in queries]  # 生成翻译

    # 计算奖励
    rewards = [compute_reward(r, ref["output"]) for r, ref in zip(responses, batch)]

    # PPO 更新
    ppo_trainer.step(queries, responses, rewards)

    print(f"Epoch {epoch}: Avg Reward = {sum(rewards) / len(rewards)}")