from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import EarlyStoppingCallback
from datasets import load_dataset
import numpy as np
import evaluate
from aspect_data import *

model = AutoModelForTokenClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=len(label_names),
    id2label=id2tag,
    label2id=tag2id

)
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

print("loaded the model...")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, tags in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = load_dataset("json", data_files={"train": "data/train/train_detection.json",
                                           "test": "data/testing/test_detection.json"})
dataset = dataset.map(tokenize_and_align_labels, batched=True)

train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
test_dataset = dataset["test"]

print("split the data...")

metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_names[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    return metric.compute(predictions=true_predictions, references=true_labels)


training_args = TrainingArguments(
    output_dir="aspect_model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_overall_accuracy',
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

trainer.train()
print("the model is trained")
results = trainer.evaluate(test_dataset)
print(results)

"""
{'eval_loss': 0.3414037823677063, 'eval_ACTING': {'precision': 0.5161290322580645, 'recall': 0.8888888888888888, 'f1': 0.6530612244897959, 'number': 18}, 'eval_HUMOR': {'precision': 0.9444444444444444, 're
call': 1.0, 'f1': 0.9714285714285714, 'number': 17}, 'eval_PICTURE': {'precision': 0.782608695652174, 'recall': 1.0, 'f1': 0.878048780487805, 'number': 18}, 'eval_PLOT': {'precision': 0.7142857142857143, '
recall': 0.7894736842105263, 'f1': 0.7500000000000001, 'number': 19}, 'eval_SOUND': {'precision': 0.7777777777777778, 'recall': 1.0, 'f1': 0.8750000000000001, 'number': 21}, 'eval_overall_precision': 0.725
, 'eval_overall_recall': 0.9354838709677419, 'eval_overall_f1': 0.8169014084507041, 'eval_overall_accuracy': 0.9370967741935484, 'eval_runtime': 3.5025, 'eval_samples_per_second': 28.266, 'eval_steps_per_second': 1.142, 'epoch': 10.0}
"""
