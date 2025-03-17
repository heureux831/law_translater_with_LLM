from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import pandas as pd

class RewardDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["score"], dtype=torch.float)
        }

class RewardModelTrainer:
    def __init__(self, model_path, train_data):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16
        )
        self.train_data = train_data
        
    def prepare_dataset(self):
        dataset = Dataset.from_pandas(pd.DataFrame(self.train_data))
        return RewardDataset(dataset, self.tokenizer)
    
    def train(self):
        training_args = TrainingArguments(
            output_dir="./reward_model",
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            save_steps=1000,
            logging_steps=100,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.prepare_dataset(),
        )
        
        trainer.train()
        trainer.save_model("./reward_model_final")
        
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = outputs.logits[0].item()
            
        return reward 