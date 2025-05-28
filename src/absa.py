from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
absa_model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")

nlp = spacy.load("ru_core_news_sm")

model = SentenceTransformers('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

aspect_keywords = {
    "plot": ["сюжет", "история", "повествование"],
    "acting": ["актёр", "актриса", "игра", "исполнение"],
    "humor": ["юмор", "шутка", "смешно"],
    "picture": ["оператор", "съёмка", "визуал", "картинка"],
    "sound": ["музыка", "звуковое сопровождение", "саундтрек"]
}

aspect_descriptions = {
    "plot": "сюжет фильма, история, повествование",
    "acting": "игра актёров, актёрское мастерство",
    "humor": "шутки, юмор, смешные сцены",
    "cinematography": "визуальный стиль, съёмка, картинка",
    "soundtrack": "музыка, аудио, звуковое сопровождение"
}

aspect_embeddings = {
    aspect: model.encode(desc, convert_to_tensor=True)
    for aspect, desc in aspect_descriptions.items()
}


def analyze_film(reviews):
    aspects = [0, 0, 0, 0, 0]

    for review in reviews:
        aspects+=analyze_review(review)

    return aspects


def module_of_number(num):
    if num>0: return 1
    if num<0: return -1
    return 0


def is_in_sentence(aspect, lemmatized_sentence):
    for aspect_keyword in aspect_keywords[aspect]:
        for word in lemmatized_sentence:
            if word==aspect_keyword:
                return 1
    return 0


def analyze_review(review):


    nltk.download('punkt_tab', quiet=True)

    sentences = sent_tokenize(review)

    aspect_scores = {
        "humor": 0,
        "acting": 0,
        "plot": 0,
        "picture": 0,
        "sound": 0
    }

    for sentence in sentences:

        #check for implicit usage
        best_aspect = detect_implicit_aspect(sentence, aspect_embeddings)
        if best_aspect: continue


        #check if explicitly mentioned -> get aspect score
        doc = nlp(sentence)
        lemmatized_tokens = [token.lemma_ for token in doc]

        for aspect in aspect_keywords.keys():
            print(aspect)
            if is_in_sentence(aspect, lemmatized_tokens):
                aspect_scores[aspect]+=absa_by_aspect(sentence, aspect)


    normalized_aspect_scores = [module_of_number(x) for x in aspect_scores.values()]
    return normalized_aspect_scores


def absa_by_aspect(sentence, aspect):
    sm = 0
    print("*")
    for word in aspect_keywords[aspect]:

        inputs = tokenizer(f"[CLS] {sentence} [SEP] {word} [SEP]",
                           return_tensors="pt",
                           truncation=True,
                           max_length=512)
        if inputs['input_ids'].shape[1]>512:
            return 0
        outputs = absa_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        probs = probs.detach().numpy()[0]
        ind_mx = np.argmax(probs)

        sm += [-1, 0, 1][ind_mx]
        print(sm)

    return module_of_number(sm)


def detect_implicit_aspect(sentence, aspect_embedding, threshold=0.4):
    embedding = model.encode(sentence, convert_to_tensor=True)
    scores = {
        aspect: util.cos_sim(embedding, asp_emb)
        for aspect, asp_emb in aspect_embedding.items()
    }
    best_aspect, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score<=threshold:
        return best_aspect
    return None



if __name__ == "__main__":
    review="Сюжет был интересным. А вот актеры плохо сыграли."
    print(analyze_review(review))
