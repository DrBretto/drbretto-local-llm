import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Load the JSON data
with open('data/sentiment_score/sentiment_scores.json', 'r') as file:
    data = json.load(file)

# Split the data into training and validation
train_data = data[:int(len(data) * 0.8)]
val_data = data[int(len(data) * 0.8):]

# Define a custom dataset class
class ArticleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])  # Model expects labels to be a tensor
        return item

    def __len__(self):
        return len(self.labels)

# Load tokenizer and tokenize the articles
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Tokenize articles
train_encodings = tokenizer([item['article'] for item in train_data], truncation=True, padding=True, return_tensors="pt")
val_encodings = tokenizer([item['article'] for item in val_data], truncation=True, padding=True, return_tensors="pt")

# Prepare the datasets
train_labels = [item['score'] for item in train_data]  # Scores are already in the correct format
val_labels = [item['score'] for item in val_data]

train_dataset = ArticleDataset(train_encodings, train_labels)
val_dataset = ArticleDataset(val_encodings, val_labels)

# Load model
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model.to('cuda')  # If you're using a GPU

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./your_model_directory")
tokenizer.save_pretrained("./your_model_directory")
