# This file contains the implementation of baseline T5 classifier.

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

labeled_reviews = pd.read_csv('../datafiles/labeled_reviews.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length = 512
model_name='t5-base'

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs.input_ids.flatten(),
            'attention_mask': inputs.attention_mask.flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }
    
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForSequenceClassification.from_pretrained(model_name)
model.to(device)

test_size=0.2
random_state=42
learning_rate=2e-5
batch_size=16
X = labeled_reviews['clean_content'].to_list()
y = labeled_reviews['Privacy-related'].to_list()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=test_size, random_state=random_state
)

train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=learning_rate)

def train():
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader)

    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / len(train_loader)})

    avg_loss = total_loss / len(train_loader)
    print(f'Average Loss: {avg_loss:.4f}')

def evaluate():
    model.eval()
    predictions = []
    pred_probs = []

    for text in X_test:
        inputs = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prob = torch.softmax(outputs.logits, dim=-1)[:, 1].item()
        pred = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred)
        pred_probs.append(prob)
    
    return predictions, pred_probs

train()
predictions, pred_probs = evaluate()
results = classification_report(y_test, predictions, output_dict=True)
print(results) # Report the results.