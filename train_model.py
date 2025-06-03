import huggingface_hub
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
dataset = load_dataset("csv", data_files={"train": "train.csv", "test":"test.csv"})
absa_model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment", num_labels=15)




def tokenize(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)



args = TrainingArguments(
    output_dir="absa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=absa_model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()
