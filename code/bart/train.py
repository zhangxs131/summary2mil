from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import torch
import pandas as pd

train_data_path="../../data/dataset/train.json"
val_data_path="../../data/dataset/validation.json"
pretrain_model="../../pretrain_model/bart-base"

class SummaryDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_input_len = 512
        self.max_output_len = 150

        if data_path.split('.')[-1]=='csv':
            df=pd.read_csv(data_path)
        elif data_path.split('.')[-1]=='json':
            df = pd.read_json(data_path)

        self.data=[(data['中文标题'],data['整编内容']) for i,data in df.iterrows()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        input_ids = self.tokenizer.encode(line[1], max_length=self.max_input_len, truncation=True, padding='max_length',
                                          return_tensors='pt')
        output_ids = self.tokenizer.encode(line[0], max_length=self.max_output_len, truncation=True,
                                           padding='max_length', return_tensors='pt')
        return {'input_ids': input_ids.squeeze(), 'attention_mask': input_ids.squeeze().gt(0),
                'decoder_input_ids': output_ids.squeeze()[:-1],
                'decoder_attention_mask': output_ids.squeeze().gt(0)[:-1], 'labels': output_ids.squeeze()[1:]}

tokenizer = BartTokenizer.from_pretrained(pretrain_model)
model = BartForConditionalGeneration.from_pretrained(pretrain_model)

# 初始化训练和验证数据集
train_dataset = SummaryDataset(tokenizer, train_data_path)
val_dataset = SummaryDataset(tokenizer, val_data_path)


# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=1000,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
