from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

# 加载数据集
dataset = load_dataset("csv", data_files={"train": "../../data/train.csv", "validation": "../../data/dev.csv"})

# 初始化 tokenizer 和模型
tokenizer = BartTokenizer.from_pretrained('./bart-base')
model = BartForConditionalGeneration.from_pretrained('./bart-base')

# 预处理数据集
def preprocess_function(examples):
    inputs = [doc for doc in examples['content_cn']]
    targets = [summ for summ in examples['title']]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

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
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
