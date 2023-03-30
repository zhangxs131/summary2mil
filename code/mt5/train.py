from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TextDataset, DataCollatorForSeq2Seq, Trainer, TrainingArguments

# 加载模型和 tokenizer
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')

# 定义原文和摘要的文件路径
train_file_path = 'train.txt'
val_file_path = 'val.txt'

# 定义模板和需要替换的关键词
template = "在本次比赛中，中国选手 <extra_id_0> 获得了 <extra_id_1> ，祝贺她！"
keyword = "<extra_id_0>"

# 定义数据集和数据处理器
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file_path,
    block_size=512,
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=val_file_path,
    block_size=512,
)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=500,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.01,
)

# 定义微调函数
def model_template_finetuning(template, keyword, train_dataset, val_dataset, data_collator, model, tokenizer, training_args):
    # 根据模板和关键词生成训练数据
    train_examples = []
    with open(train_dataset.file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            text, label = line.split('\t')
            train_examples.append((template.replace(keyword, text), label))
    val_examples = []
    with open(val_dataset.file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            text, label = line.split('\t')
            val_examples.append((template.replace(keyword, text), label))

    # 定义训练器并开始微调
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        examples=train_examples,
        block_size=512,
    )
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        examples=val_examples,
        block_size=512,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    return model

# 执行微调函数
model = model_template_finetuning(template, keyword, train_dataset, val_dataset, data_collator, model, tokenizer, training_args)
