from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments


model = AutoModelForTokenClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=3)  # B/I/O
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

training_args = TrainingArguments(
    output_dir="./aspect_model",
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
