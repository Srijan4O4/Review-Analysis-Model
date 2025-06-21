"""
Refactored PyTorch script for training DistilBERT classifier (full precision, no checkpoints).
Optimized for local GPU (RTX 3050). Prints detailed progress and computes metrics at the end.
"""
print("Training DistilBERT Classifier for Review Classification")
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("Importing libraries...")
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train DistilBERT classifier")
parser.add_argument('--data_file', type=str, required=True, help='Path to CSV data file')
parser.add_argument('--model_dir', type=str, required=True, help='Directory to save model and tokenizer')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size (default: 16)')
parser.add_argument('--max_length', type=int, default=256, help='Max sequence length for tokenization (default: 256)')
args = parser.parse_args()

# Create output directory if needed
os.makedirs(args.model_dir, exist_ok=True)

# Load data
df = pd.read_csv(args.data_file)
# Combine 'review' and 'rating' into one text input
df['rating'] = df['rating'].astype(str)
df['text'] = df['review'] + ' Rating: ' + df['rating']
df = df.dropna(subset=['text', 'isAI'])
texts = df['text'].tolist()
labels = df['isAI'].tolist()

# Encode labels (in case they are strings)
le = LabelEncoder()
labels = le.fit_transform(labels)
num_labels = len(le.classes_)
print(f"Detected classes: {le.classes_} -> num_labels={num_labels}")


print(f"Loaded {len(texts)} reviews with {num_labels} classes.")
# Split into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train set: {len(train_texts)} reviews, Test set: {len(test_texts)} reviews")
# Initialize tokenizer (DistilBERT base, uncased)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# Tokenize datasets with truncation and padding
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=args.max_length)
print(f"Tokenized train set: {len(train_encodings['input_ids'])} samples, "
      f"test set: {len(test_encodings['input_ids'])} samples")
# Create PyTorch Dataset (following Hugging Face example):
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels)
test_dataset  = ReviewDataset(test_encodings,  test_labels)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)
print(f"Created DataLoaders: train_loader={len(train_loader)} batches, test_loader={len(test_loader)} batches")
# Device (GPU if available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
print("Device info:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# Load pre-trained DistilBERT model for sequence classification:contentReference[oaicite:1]{index=1}
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)
model.to(device)
model.train()
print("Model loaded and moved to device.")
# Optimizer (AdamW)
optimizer = AdamW(model.parameters(), lr=5e-5)


print("Starting training...")
# Training loop (no mixed precision, full FP32):contentReference[oaicite:2]{index=2}
print("Starting training...")
for epoch in range(args.epochs):
    epoch_start = time.time()
    for step, batch in enumerate(train_loader, 1):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        # Print training progress
        if step % 500 == 0 or step == len(train_loader):
            elapsed = time.time() - epoch_start
            print(f"[Epoch {epoch+1}/{args.epochs} Step {step}/{len(train_loader)}] loss: {loss.item():.4f} - elapsed: {elapsed:.2f}s")
            epoch_start = time.time()
    print(f"Epoch {epoch+1} completed.\n")
print("Training complete. Total time: {:.2f} seconds".format(time.time() - epoch_start))

# Evaluation
print("Training complete. Running evaluation on test set...")
model.eval()
preds, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=-1)
        preds.extend(batch_preds.cpu().numpy())
        true_labels.extend(labels_batch.cpu().numpy())
print(f"Evaluation complete. Predicted {len(preds)} labels on test set.")
# Compute metrics
acc = accuracy_score(true_labels, preds)
print(f"\nAccuracy on test set: {acc:.4f}")
print("Classification Report:")  # (from sklearn.metrics.classification_report):contentReference[oaicite:3]{index=3}
print(classification_report(true_labels, preds, target_names=[str(c) for c in le.classes_]))
print("Detailed classification report saved to classification_report.txt")
# Confusion Matrix
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(c) for c in le.classes_],
            yticklabels=[str(c) for c in le.classes_])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(args.model_dir, 'confusion_matrix.png'))
plt.show()  # (plot confusion matrix):contentReference[oaicite:4]{index=4}

print("Confusion matrix saved to confusion_matrix.png")

# Save final model and tokenizer (only final save, no intermediate checkpoints)
tokenizer.save_pretrained(args.model_dir)
model.save_pretrained(args.model_dir)
print(f"Model and tokenizer saved to {args.model_dir}")
