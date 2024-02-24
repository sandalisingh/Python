import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the emotion classification model using BERT
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Function to tokenize and encode text data
def preprocess_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    return inputs

# Example emotion classification dataset (replace this with your dataset)
data = pd.DataFrame({
    'text': ['I feel so happy today!', 
    'I am sad because of the news.', 
    'This is so exciting!', 
    'Feeling calm and relaxed.'],
    'emotion': ['happy', 'sad', 'excited', 'calm']
})

# Encode emotion labels
label_map = {label: idx for idx, label in enumerate(data['emotion'].unique())}
data['label'] = data['emotion'].map(label_map)

# Split dataset into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Tokenize and encode text data
train_inputs = preprocess_text(train_texts.tolist())
test_inputs = preprocess_text(test_texts.tolist())

# Convert labels to PyTorch tensors
train_labels = torch.tensor(train_labels.tolist())
test_labels = torch.tensor(test_labels.tolist())

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize and train the EmotionClassifier model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionClassifier(num_classes=len(label_map)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(3):  # Adjust number of epochs as needed
    model.train()
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        logits = model(input_ids, attention_mask)
        _, predicted = torch.max(logits, 1)
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)
