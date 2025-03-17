# %% [markdown]
# # 划分数据集

# %%
import json

def split_json(input_file: str, output_file: str, remaining_file: str, num_items: int = 700):
    """
    从 JSON 文件中选取指定数量的数据转移到另一个 JSON 文件，并将剩余数据存入一个新的 JSON 文件。
    
    :param input_file: 输入 JSON 文件路径
    :param output_file: 输出 JSON 文件路径（存放选取的数据）
    :param remaining_file: 剩余数据存放的 JSON 文件路径
    :param num_items: 要选取的数据数量，默认为 700
    """
    try:
        # 读取 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据是列表类型
        if not isinstance(data, list):
            raise ValueError("JSON 文件的根对象应为列表！")
        
        # 选取前 num_items 项数据，并获取剩余数据
        selected_data = data[:num_items]
        remaining_data = data[num_items:]
        
        # 写入新的 JSON 文件（存放选取的数据）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_data, f, ensure_ascii=False, indent=4)
        
        # 写入剩余数据到新的 JSON 文件
        with open(remaining_file, 'w', encoding='utf-8') as f:
            json.dump(remaining_data, f, ensure_ascii=False, indent=4)
        
        print(f"成功写入 {len(selected_data)} 项数据到 {output_file}")
        print(f"剩余 {len(remaining_data)} 项数据存入 {remaining_file}")
    except Exception as e:
        print(f"处理文件时出错: {e}")


split_json('/root/nas-private/Qwen2.5-sft/train.json', '/root/nas-private/Qwen2.5-sft/test.json', '/root/nas-private/Qwen2.5-sft/train.json')
