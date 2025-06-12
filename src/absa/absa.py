from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from train_absa_model import model, tokenizer
from aspect_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "train_model/absa_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
absa_model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


def predict_aspect_sentiment(text, aspect):
    inputs = tokenizer(f"{aspect} [SEP] {text}", return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_id = logits.argmax().item()
    full_label = id_to_label[predicted_class_id]
    if predicted_class_id %3==0: return 1
    elif predicted_class_id%3==1: return 0
    return -1


def predict_aspects(sentence):
    pass

def analyze_review(review):

    aspect_output = {
        "plot": 0,
        "acting": 0,
        "humor": 0,
        "picture": 0,
        "sound": 0
    }

    for sentence in sent_tokenize(review):
        predicted_aspects = predict_aspects(sentence)

        for aspect in predicted_aspects:
            val = predict_aspect_sentiment(sentence, aspect)
            aspect_output[aspect]+=val

    return list(aspect_output.items())




if __name__ == "__main__":
    review = "Сюжет был интересным. А вот актеры плохо сыграли. Было смешно. "
    print(analyze_review(review))