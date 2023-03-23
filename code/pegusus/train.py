import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import PegasusForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from tokenizers_pegasus import PegasusTokenizer

# 定义训练和验证数据的路径
train_data_path = "../../data/train.csv"
val_data_path = "../../data/dev.csv"
pretrain_model_path="../../pretrain_model/pegasus-238M"

# 初始化tokenizer和model
tokenizer = PegasusTokenizer.from_pretrained(pretrain_model_path)
model = PegasusForConditionalGeneration.from_pretrained(pretrain_model_path)


# 定义训练和验证数据的Dataset
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


# 初始化训练和验证数据集
train_dataset = SummaryDataset(tokenizer, train_data_path)
val_dataset = SummaryDataset(tokenizer, val_data_path)
print(len(train_dataset))


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #if True:
        # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# 初始化训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=1000,
    save_total_limit=5,
    save_strategy="epoch",
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# 开始训练模型
trainer.train()
