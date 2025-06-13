from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import numpy as np
import evaluate
from aspect_data import label_map

dataset = load_dataset("csv", data_files={"train": "movie_train_absa/train.csv", "test": "data/testing/test_absa.csv"})

num_labels = len(label_map)
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

print("loaded the model...")


def encode_labels(example):
    aspect_sentiment = f"{example['aspect']}_{example['sentiment']}"
    example["label"] = label_map[aspect_sentiment]
    return example


def tokenize(examples):
    texts = [f"{sentence} [SEP] {aspect}" for sentence, aspect in zip(examples["sentence"], examples["aspect"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=32, return_special_tokens_mask=True)


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.map(encode_labels)
dataset = dataset.remove_columns(["sentence", "aspect", "sentiment"])

columns_to_keep = ["input_ids", "attention_mask", "label"]
for split in dataset.keys():
    cols_to_remove = [col for col in dataset[split].column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(cols_to_remove)
train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)

train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
test_dataset = dataset["test"]

print("split the data...")

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=predictions, references=labels)["accuracy"],
        "f1_macro": metric_f1.compute(predictions=predictions, references=labels, average="macro")['f1']
    }


args = TrainingArguments(
    output_dir="absa_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=7,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
print("model is trained.")
results = trainer.evaluate(test_dataset)
print(results)
