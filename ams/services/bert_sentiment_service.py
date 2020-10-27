import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


def inference_model(model, data_loader, device, n_examples):
    model = model.eval()

    all_preds = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        _, preds = torch.max(outputs, dim=1)
        all_preds.append(preds)

    return all_preds


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        #     self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #     output = self.drop(pooled_output)
        output = pooled_output
        return self.out(output)


class GPReviewDataset(Dataset):

    def __init__(self, tweets, tokenizer, max_len):
        self.tweets = tweets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        print(f"Item: {item}")

        tweet = str(self.tweets[item])
        print(f"Encoding tweet {tweet[:20]}...")

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'tweet': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        tweets=df["text"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

    # train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    # val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)


def get_bert_preds(df: pd.DataFrame):
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'

    BATCH_SIZE = 15
    MAX_LEN = 280

    print(f'df_red count: {df.shape[0]}')
    print(f"df_red cols: {df.columns}")
    val_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    encoding = tokenizer.encode_plus(
        sample_txt,
        max_length=MAX_LEN,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
    )

    print(encoding.keys())

    last_hidden_state, pooled_output = bert_model(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask']
    )

    device = torch.device("cpu")
    preds = inference_model(bert_model, val_data_loader, device, df.shape[0])

    print(f"Preds type: {type(preds[0])}")
    print(f"Length preds: {len(preds)}")
    print(f"Length preds: {preds}")
