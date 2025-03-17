
# %% [markdown]
# # 导入环境

# %%
import os
# 使用一张卡训练 避免出现chunk错误
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

#%%
train_data_path = '/root/nas-private/Qwen2.5-sft/train.json'
model_path = '/root/nas-private/Qwen2.5-sft/model/Qwen2.5-7B-Instruct'
tokenizer_path = '/root/nas-private/Qwen2.5-sft/model/Qwen2.5-7B-Instruct'
lora_weights_path = "./output/Qwen2.5_7b_instruct_lora_v1"

# %%
# 将JSON文件转换为CSV文件
df = pd.read_json(train_data_path)
ds = Dataset.from_pandas(df)

# %%
ds[:3]

# %% [markdown]
# # 处理数据集

# %%
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)
tokenizer

# %%
def process_func(example):
    MAX_LENGTH = 384  # Llama 分词器会将一个中文字切分为多个 token，因此需要放开一些最大长度，保证数据的完整性

    messages = [
        {"role": "system", "content": "你擅长翻译中日法律文本"},
        {"role": "user", "content": example['instruction'] + example['input']},
        {"role": "assistant", "content": example['output']}
    ]
    
    # 使用 tokenizer.apply_chat_template 生成 token 序列
    tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")

    input_ids = tokenized.squeeze(0).tolist()  # 转换为列表
    attention_mask = [1] * len(input_ids)  # 由于 apply_chat_template 已经构造完整输入，直接设置为 1
    
    # 计算 labels，assistant 之前的 token 设为 -100
    system_user_tokenized = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, return_tensors="pt")
    prefix_length = system_user_tokenized.shape[1]
    labels = [-100] * prefix_length + input_ids[prefix_length:]

    # 截断到 MAX_LENGTH
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# %%
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id



# %%
tokenizer.decode(tokenized_id[0]['input_ids'])

# %%
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

# %% [markdown]
# # 创建模型

# %%
import torch

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda:0")
model

# %%
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

# %%
model.dtype

# %%
model.device

# %% [markdown]
# # lora 

# %%
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config

# %%
model = get_peft_model(model, config)
config

# %%
model.device

# %%
model.print_trainable_parameters()

# %% [markdown]
# # 配置训练参数

# %%
args = TrainingArguments(
    output_dir=lora_weights_path,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    run_name="lora_700test_6560train_v1"
)

# %%

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# %%

trainer.train()


# 7034, 'grad_norm': 2.270235538482666, 'learning_rate': 1.4634146341463415e-05, 'epoch': 1.71}                                    {'loss': 0.7442, 'grad_norm': 2.6562812328338623, 'learning_rate': 1.2195121951219513e-05, 'epoch': 1.76}                                   
# {'loss': 0.7775, 'grad_norm': 2.636554002761841, 'learning_rate': 9.756097560975611e-06, 'epoch': 1.8}                                      {'loss': 0.7706, 'grad_norm': 2.3621671199798584, 'learning_rate': 7.317073170731707e-06, 'epoch': 1.85}                                    {'loss': 0.7716, 'grad_norm': 2.217282295227051, 'learning_rate': 4.8780487804878055e-06, 'epoch': 1.9}                                     {'loss': 0.7366, 'grad_norm': 2.1729061603546143, 'learning_rate': 2.4390243902439027e-06, 'epoch': 1.95}                                   {'loss': 0.7356, 'grad_norm': 2.2804503440856934, 'learning_rate': 0.0, 'epoch': 2.0}