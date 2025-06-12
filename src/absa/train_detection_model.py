from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import EarlyStoppingCallback
from datasets import load_dataset
import numpy as np
import evaluate




label_names = ["O", "B-PLOT", "I-PLOT", "B-ACTING", "I-ACTING", "B-PICTURE", "I-PICTURE", "B-HUMOR", "I-HUMOR"]
tag2id = {tag: i for i, tag in enumerate(label_names)}
id2tag = {i: tag for i, tag in enumerate(label_names)}

model = AutoModelForTokenClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=3,
    id2label=id2tag,
    label2id=tag2id

)
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

print("loaded the model...")

def tokenize_and_align_labels(examples):
    tokenized_inputs= tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels=[]
    for i, tags in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids=[]
        previous_word_idx =None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx!= previous_word_idx:
                label_ids.append(tag2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = load_dataset("json", data_files={"train": "../movie_train_absa/train_detection.json", "test":"data/testing/test_detection.json"})
dataset = dataset.map(tokenize_and_align_labels, batched=True)

train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset=train_test_split["train"]
eval_dataset=train_test_split["test"]
test_dataset = dataset["test"]

print("split the data...")

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_names[l] for l in label if l!=-100] for label in labels]
    true_predictions = [[label_names[p] for (p, l) in zip(prediction, label) if l!=-100] for prediction, label in zip(predictions, labels)]
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
    metric_for_best_model='f1',
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

