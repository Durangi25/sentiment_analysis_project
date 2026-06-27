import numpy as np
import pandas as pd
import re
import string
import pickle
from pathlib import Path
from collections import Counter

from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer

ps = PorterStemmer()
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'static' / 'model'
MODEL_FILE = MODEL_DIR / 'model.pickle'
VOCAB_FILE = MODEL_DIR / 'vocabulary.txt'
DATA_FILE = BASE_DIR / 'artifacts' / 'sentiment_analysis.csv'
STOPWORDS_FILE = MODEL_DIR / 'corpora' / 'stopwords' / 'english'

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_stopwords():
    if STOPWORDS_FILE.exists():
        return STOPWORDS_FILE.read_text(encoding='utf-8').splitlines()
    return []

sw = load_stopwords()


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data['tweet'] = data['tweet'].apply(remove_punctuations)
    data["tweet"] = data["tweet"].str.replace(r'\d+', '', regex=True)
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(x for x in x.split() if x not in sw))
    data['tweet'] = data['tweet'].apply(lambda x: ' '.join(ps.stem(x) for x in x.split()))
    return data['tweet']


def build_vocabulary(sentences, min_freq=10):
    counter = Counter()
    for sentence in sentences:
        counter.update(str(sentence).lower().split())
    return [token for token, count in counter.items() if count > min_freq]


def save_vocabulary(tokens, filename):
    filename.write_text('\n'.join(tokens), encoding='utf-8')


def load_vocabulary():
    if VOCAB_FILE.exists():
        return VOCAB_FILE.read_text(encoding='utf-8').splitlines()
    if DATA_FILE.exists():
        texts, _ = load_dataset()
        tokens = build_vocabulary(texts)
        save_vocabulary(tokens, VOCAB_FILE)
        return tokens
    return []


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_FILE}')
    data = pd.read_csv(DATA_FILE)
    if 'tweet' not in data.columns or 'label' not in data.columns:
        raise ValueError('Dataset must contain tweet and label columns')
    data = data.dropna(subset=['tweet', 'label'])
    texts = data['tweet'].astype(str).tolist()
    labels = data['label'].astype(int).tolist()
    return texts, labels


def vectorizer(ds, vocabulary=None):
    vocabulary = tokens if vocabulary is None else vocabulary
    vectorized_lst = []
    for sentence in ds:
        sentence_words = set(str(sentence).split())
        sentence_lst = np.zeros(len(vocabulary), dtype=np.float32)
        for i, token in enumerate(vocabulary):
            if token in sentence_words:
                sentence_lst[i] = 1
        vectorized_lst.append(sentence_lst)
    return np.asarray(vectorized_lst, dtype=np.float32)


def train_model():
    texts, labels = load_dataset()
    new_tokens = build_vocabulary(texts)
    save_vocabulary(new_tokens, VOCAB_FILE)
    global tokens
    tokens = new_tokens
    vectorized = vectorizer(texts, tokens)
    model = LogisticRegression(max_iter=1000)
    model.fit(vectorized, labels)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    return model


def load_model():
    if MODEL_FILE.exists() and VOCAB_FILE.exists():
        try:
            with open(MODEL_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return train_model()


tokens = load_vocabulary()
model = load_model()


POSITIVE_KEYWORDS = {
    'good', 'great', 'excellent', 'awesome', 'love', 'nice', 'happy',
    'best', 'amazing', 'fantastic', 'wonderful', 'pleasant', 'enjoy',
    'like', 'liked'
}

NEGATIVE_KEYWORDS = {
    'bad', 'terrible', 'awful', 'hate', 'worst', 'sad', 'horrible',
    'stupid', 'problem', 'angry', 'disappoint', 'disappointed', 'poor'
}


NEGATION_WORDS = {'not', 'never', 'no', 'none', "isn't", "wasn't", "didn't", "don't", "doesn't"}


def get_prediction(vectorized_text, raw_text=None):
    if raw_text is not None:
        normalized = raw_text.strip().lower()
        text_tokens = re.findall(r"\b\w+\b", normalized)
        text_set = set(text_tokens)

        if text_set & NEGATION_WORDS:
            if text_set & POSITIVE_KEYWORDS:
                return 'negative'
            if text_set & NEGATIVE_KEYWORDS:
                return 'positive'

        positive_matches = text_set & POSITIVE_KEYWORDS
        negative_matches = text_set & NEGATIVE_KEYWORDS

        if positive_matches and not negative_matches:
            return 'positive'
        if negative_matches and not positive_matches:
            return 'negative'
        if positive_matches and negative_matches:
            return 'negative'

    prediction = model.predict(vectorized_text)
    prediction = prediction[0] if hasattr(prediction, '__len__') else prediction
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(vectorized_text)[0]

    if probability is not None and max(probability) < 0.75 and raw_text is not None:
        text_tokens = set(re.findall(r"\b\w+\b", raw_text.lower()))
        if text_tokens & POSITIVE_KEYWORDS:
            return 'positive'
        if text_tokens & NEGATIVE_KEYWORDS:
            return 'negative'

    if prediction == 1:
        return 'negative'
    return 'positive'