from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment", num_labels=15)

dataset = load_dataset("csv", data_files={"train": "train.csv", "test":"test_absa.csv"})

label_map = {
    "plot_positive": 0,
    "plot_neutral": 1,
    "plot_negative": 2,
    "acting_positive": 3,
    "acting_neutral": 4,
    "acting_negative": 5,
    "picture_positive": 6,
    "picture_neutral": 7,
    "picture_negative": 8,
    "sound_positive": 9,
    "sound_neutral": 10,
    "sound_negative": 11,
    "humor_positive": 12,
    "humor_neutral": 13,
    "humor_negative": 14
}

def encode_labels(example):
    aspect_sentiment = f"{example['aspect']}_{example['sentiment']}"
    example["label"] = label_map[aspect_sentiment]
    return example

dataset = dataset.map(encode_labels)

def tokenize(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["input_ids", "attention_mask", "label"]])

dataset["train"] = dataset["train"].shuffle(seed=42)
dataset["test"] = dataset["test"].shuffle(seed=42)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

args = TrainingArguments(
    output_dir="absa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()
