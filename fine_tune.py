import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Assuming your dataset is in a CSV file named 'dataset.csv' with columns 'article' and 'score'
dataset_path = 'path/to/your/dataset.csv'  # Update this path

# Load the dataset
raw_datasets = load_dataset('csv', data_files=dataset_path, split='train')
raw_datasets = raw_datasets.train_test_split(test_size=0.1)  # Splitting the dataset

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['article'], padding="max_length", truncation=True)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # Adjust model size as needed

# Tokenize the dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Load the model
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to("cuda")  # Adjust model size as needed

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,  # Adjust as needed
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("./your_model_directory")
tokenizer.save_pretrained("./your_model_directory")
