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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
absa_model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment").to(device)

nlp = spacy.load("ru_core_news_sm")

sim_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

aspect_descriptions = {
    "plot": ["сюжет", "история", "повествование", "сценарий", "нарратив"],
    "acting": ["актёр", "актриса", "игра", "исполнение", "актёрский состав"],
    "humor": ["юмор", "шутка", "смешно", "комедийный", "сатира"],
    "picture": ["оператор", "съёмка", "визуал", "картинка", "спецэффекты"],
    "sound": ["музыка", "звуковое сопровождение", "саундтрек", "озвучка", "звуковые эффекты"]
}

aspect_embeddings = {
    aspect: torch.mean(sim_model.encode(desc, convert_to_tensor=True), dim=0)
    for aspect, desc in aspect_descriptions.items()
}

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for aspect, keywords in aspect_descriptions.items():
    patterns = [nlp.make_doc(keyword) for keyword in keywords]
    matcher.add(aspect, None, *patterns)

def analyze_film(reviews):
    aspect_scores = defaultdict(list)
    for review in reviews:
        scores = analyze_review(review)
        for aspect, score_list in scores.items():
            aspect_scores[aspect].extend(score_list)
    return {k: sum(v)/len(v) if v else 0 for k, v in aspect_scores.items()}

def analyze_review(review):
    doc = nlp(review)
    aspect_scores = defaultdict(list)
    for chunk in split_into_chunks(doc, max_chars=2000):
        chunk_doc = nlp(chunk.text)
        explicit_aspects = find_explicit_aspects(chunk_doc)
        for sent in chunk_doc.sents:
            implicit_aspect = detect_implicit_aspect(sent.text, aspect_embeddings)
            if implicit_aspect:
                explicit_aspects[implicit_aspect].append(implicit_aspect)
        for aspect, matches in explicit_aspects.items():
            aspect_term = select_representative_term(matches)
            score = absa_by_aspect(sent.text, aspect)
            aspect_scores[aspect].append(score)
    return aspect_scores

def select_representative_term(terms):
    return max(terms, key=len) if terms else ""

def split_into_chunks(doc, max_chars=2000):
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in doc.sents:
        sent_len = len(sent.text)
        if current_len + sent_len > max_chars and current_chunk:
            chunks.append(nlp(" ".join(current_chunk)))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent.text)
        current_len += sent_len
    if current_chunk:
        chunks.append(nlp(" ".join(current_chunk)))
    return chunks

def find_explicit_aspects(doc):
    aspects = defaultdict(list)
    matches = matcher(doc)
    for match_id, start, end in matches:
        aspect = nlp.vocab.strings[match_id]
        aspects[aspect].append(doc[start:end].text)
    return aspects

def absa_by_aspect(sentence, aspect):
    scores = []
    for word in aspect_descriptions.get(aspect, []):
        try:
            inputs = tokenizer(
                f"[CLS] {sentence} [SEP] {word} [SEP]",
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = absa_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            sentiment_idx = torch.argmax(probs).item()
            scores.append([-1, 0, 1][sentiment_idx])
        except Exception as e:
            continue
    return sum(scores) / len(scores) if scores else 0


def detect_implicit_aspect(sentence, aspect_embedding, threshold=0.65):
    embedding = sim_model.encode(sentence, convert_to_tensor=True)

    # Ensure embedding is 2D: (1, embedding_dim)
    if len(embedding.shape) == 1:
        embedding = embedding.unsqueeze(0)

    scores = {}
    for aspect, asp_emb in aspect_embedding.items():
        # Ensure asp_emb is also 2D
        if len(asp_emb.shape) == 1:
            asp_emb = asp_emb.unsqueeze(0)
        score = util.cos_sim(embedding, asp_emb).item()
        scores[aspect] = score

    best_aspect, best_score = max(scores.items(), key=lambda x: x[1])
    return best_aspect if best_score >= threshold else None


if __name__ == "__main__":
    review = "Сюжет был интересным. А вот актеры плохо сыграли. Было смешно. "
    print(analyze_review(review))