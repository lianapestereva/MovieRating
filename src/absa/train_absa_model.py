from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers import EarlyStoppingCallback
import numpy as np
import evaluate
from aspect_data import *
from aspect_data import label_map, translated

dataset = load_dataset("csv", data_files={"train": "data/train/train.csv", "test": "data/testing/test_absa.csv"})

num_labels = len(label_map)

config = AutoConfig.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=num_labels,
    hidden_droput_prob=0.3,
    attention_probs_dropout_prob=0.2
)

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=num_labels,
    problem_type="single_label_classification"
)

print("loaded the model...")


def encode_labels(example):
    aspect_sentiment = f"{example['aspect']}_{example['sentiment']}"
    example["label"] = label_map[aspect_sentiment]
    return example


def tokenize(examples):
    texts = [f"[ASPECT] {translated[aspect]} {tokenizer.sep_token} {sentence}" for sentence, aspect in
             zip(examples["sentence"], examples["aspect"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.map(encode_labels)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)

train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
test_dataset = dataset["test"]

print("split the data...")

print("Sample train inputs:")
print(tokenizer.decode(train_dataset[0]["input_ids"]))
label_id = train_dataset[0]["label"].item()
print("Label:", label_id, id_to_label[label_id])

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
    output_dir="absa_model_1",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=7,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
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
