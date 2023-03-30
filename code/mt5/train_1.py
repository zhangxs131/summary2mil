import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader


class TemplateDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.templates = []
        self.texts = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                text, template = line.strip().split('\t')
                self.texts.append(text)
                self.templates.append(template)

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, idx):
        template = self.templates[idx]
        text = self.texts[idx]
        inputs = self.tokenizer.batch_encode_plus(
            [template.replace('keyword', text)],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'decoder_input_ids': self.tokenizer.encode(
                'summarize: ' + text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            ).squeeze(),
            'decoder_attention_mask': torch.tensor([1] * (len(text.split()) + 2) + [0] * (self.max_length - len(text.split()) - 2))
        }


def train(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids[:, :-1],
            decoder_attention_mask=decoder_attention_mask[:, :-1]
        )

        loss = torch.nn.functional.cross_entropy(
            outputs.logits.permute(0, 2, 1),
            decoder_input_ids[:, 1:],
            ignore_index=0
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


if __name__ == '__main__':
    train_data_path = 'train.txt'
    model_name = 'google/mt5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = TemplateDataset(train_data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(5):
        train_loss = train(model, dataloader, optimizer, device)
        print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.3f}')
