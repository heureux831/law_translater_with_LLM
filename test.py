
# %% [markdown]
# # 合并加载模型

# %%
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = '/root/nas-private/Qwen2.5-sft/model/Qwen2.5-7B-Instruct'
lora_path = '/root/nas-private/Qwen2.5-sft/law-translater-qwen2.5-7b/output/Qwen2.5_7b_instruct_lora_v1/checkpoint-410' # 这里改称你的 lora 输出对应 checkpoint 地址


base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# %%
import json
import pandas as pd
import torch

# %%

test_data_path = "/root/nas-private/Qwen2.5-sft/test.json"
output_path = "/root/nas-private/Qwen2.5-sft/law-translater-qwen2.5-7b/test_output.csv"

# %%

# 读取 JSON 文件
with open(test_data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 确保 JSON 数据是列表
if not isinstance(data, list):
    raise ValueError("JSON 文件内容必须是列表形式")

# 转换为 DataFrame
df = pd.DataFrame(data)

# 确保 JSON 文件中有 'input' 字段
if 'input' not in df.columns:
    raise ValueError("JSON 文件缺少 'input' 字段")

# i = 0

# 处理每一行数据
def translate_text(example):
    messages = [
        {"role": "system", "content": "你擅长翻译中日法律文本"},
        {"role": "user", "content": example["instruction"] + example['input']}
    ]
    
    # 生成输入 token
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    prompt_length = inputs.shape[-1]

    # inputs = inputs.to(model.device)  # 确保输入和模型在相同设备上
    # print("第" + str(i) + "条数据正在测试")
    # 生成翻译结果
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=3 * prompt_length,
        temperature=0.3,
        top_p=0.8,
        # num_beams=5,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    # i = i + 1
    print(output_text)
    return output_text

# 翻译整个数据集 写入translated_text列
df['translated_text'] = df.apply(translate_text, axis=1)

# 保存结果到 CSV 文件
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"翻译完成，结果已保存到 {output_path}")