# This file contains the implementation of baseline BERT classifier.

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

labeled_reviews = pd.read_csv('../datafiles/labeled_reviews.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length = 512
num_classes = 2
model_name='bert-base-uncased'

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
class BertForSequenceClassification(nn.Module):
    def __init__(self, num_classes, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits
    
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification(num_classes, model_name)
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
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / len(train_loader)})

def evaluate():
    model.eval()
    predictions = []
    pred_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[1]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            
            predictions.extend(preds)
            pred_probs.extend(probs)
    
    return predictions, pred_probs

def train_eval(epochs=3):
    results=[]
    for epoch in range(epochs):
        train()
        predictions, pred_probs = evaluate()
        metrics = classification_report(y_test, predictions, output_dict=True)
        results.append({epoch: metrics})
    return results

results = train_eval(3)
print(results) # Report results for best epoch.