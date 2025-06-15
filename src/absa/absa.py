from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from nltk.tokenize import sent_tokenize
import torch
from src.absa.aspect_data import *

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
absa_model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\user\PycharmProjects\Movie Rating Project\absa_model\checkpoint-312")
detect_model = AutoModelForTokenClassification.from_pretrained(
    r"C:\Users\user\PycharmProjects\Movie Rating Project\aspect_model\checkpoint-150")
absa_model.eval()
detect_model.eval()


# predicts if the aspect described in a sentence positively, neutrally or negatively
def predict_aspect_sentiment(sentence, aspect):
    inputs = tokenizer(f"[ASPECT] {translated[aspect]} {tokenizer.sep_token} {sentence}", return_tensors="pt",
                       padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        absa_logits = absa_model(**inputs).logits

    predicted_class_id = absa_logits.argmax().item()
    full_label = id_to_label[predicted_class_id]

    #print(sentence, aspect, full_label)
    if predicted_class_id % 3 == 0:
        return 1
    elif predicted_class_id % 3 == 1:
        return 0
    return -1


# analyses what aspects are present in the sentence
def predict_aspects(sentence):
    inputs = tokenizer([sentence.split()], return_tensors='pt', padding=True, truncation=True, max_length=128,
                       is_split_into_words=True)
    with torch.no_grad():
        detect_logits = detect_model(**inputs).logits

    predictions = torch.argmax(detect_logits, dim=2)
    predictions = predictions[0].tolist()

    predicted_aspects = [id2tag[id_] for id_ in predictions]
    predicted_aspects_unique = set()
    for aspect in predicted_aspects:
        if aspect != 'O':
            predicted_aspects_unique.add(aspect.lower()[2:])

    return predicted_aspects_unique


def sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0


# analyses each sentence of a review and returns list of tuples aspect-overall sentiment
def analyze_review(review):
    aspect_output = {
        "сюжет": 0,
        "актерская игра": 0,
        "юмор": 0,
        "визуал": 0,
        "звук": 0
    }

    for sentence in sent_tokenize(review):
        predicted_aspects = predict_aspects(sentence)

        for aspect in predicted_aspects:
            val = predict_aspect_sentiment(sentence, aspect)
            aspect_output[translated[aspect]] += val

    return [sign(x[1]) for x in list(aspect_output.items())]


if __name__ == "__main__":
    with open(r"C:\Users\user\PycharmProjects\Movie Rating Project\data\testing\test_absa.csv", encoding="utf-8") as f:
        s = f.readline()
        for i in range(20):
            s = f.readline().split(",")[0]
            print(analyze_review(s))
