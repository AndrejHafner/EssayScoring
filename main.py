import pandas as pd
import numpy as np

from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

from utils import read_all_essays
from essay_scorers import EssayStructureScorer,EssayVocabularyScorer


def clean_data():
    cleaned_essays = []
    stemmer = PorterStemmer()
    for essay in read_all_essays():
        words = word_tokenize(essay)
        stop_words = stopwords.words('english')
        words_cleaned = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
        essay_cleaned = " ".join(words_cleaned)
        cleaned_essays.append(essay_cleaned)
    return cleaned_essays

def Tfidf_vectorize_data(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    return pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

def preprocess_data():
    essays = clean_data()
    df = Tfidf_vectorize_data(essays)
    df["unique_words"] = np.array(list(map(lambda essay: len(set(essay)),essays)))
    df["word_count"] = np.array(list(map(lambda essay: len(essay.split(" ")) ,essays)))
    return df

def cross_validate_model(model,X,y):
    scores = cross_val_score(model.classifier, X.values, np.array(y.values).ravel(), cv=5)
    print(np.average(scores))


if __name__ == "__main__":
    structure_scores = pd.read_csv("data/essays_structure_score.txt")
    vocabulary_scores = pd.read_csv("data/essays_vocabulary_score.txt")
    df = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(df, vocabulary_scores, test_size = 0.2, random_state = 42)
    voc_scorer = EssayVocabularyScorer(X_train,y_train)
    cross_validate_model(voc_scorer,X_test,y_test)

# previous best: 0.67718
# with word count = 0.73527