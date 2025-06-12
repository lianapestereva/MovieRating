from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from aspect_data import label_map

dataset = load_dataset("csv", data_files={"train": "movie_train_absa/train.csv", "test":"data/testing/test_absa.csv"})

num_labels = len(label_map)

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=num_labels, from_flax=True)
for param in model.base_model.parameters():
    param.requires_grad = False

print("loaded the model...")

def encode_labels(example):
    aspect_sentiment = f"{example['aspect']}_{example['sentiment']}"
    example["label"] = label_map[aspect_sentiment]
    return example

dataset = dataset.map(encode_labels, remove_columns=["aspect", "sentiment"])

def tokenize(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

columns_to_keep = ["input_ids", "attention_mask", "label"]
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])


full_train_dataset = dataset["train"]
train_test_split = full_train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
test_dataset = dataset["test"]

print("split the data...")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

args = TrainingArguments(
    output_dir="absa_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()
print("model is trained.")
results = trainer.evaluate(test_dataset)
print(results)