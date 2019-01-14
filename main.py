import pandas as pd
import numpy as np
import nltk
import re

from nltk import PorterStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords,brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

from utils import read_all_essays, read_essay
from essay_scorers import EssayStructureScorer, EssayVocabularyScorer, EssayMissingWordsReplacer

# nltk.download('brown')
# nltk.download('averaged_perceptron_tagger')

EXCLUDED_CLEAN_WORDS = ["lt", "gt"]
WORD_CORPUS = set(i.lower() for i in brown.words())

NOUNS = ["NN","NNS","NNP","NNPS"]
ADVERBS = ["RB","RBR","RBS"]
ADJECTIVES = ["JJ","JJR","JJS"]
VERBS = ["VB","VBD","VBG","VBN","VBP","VBZ"]




def clean_data():
    cleaned_essays = []
    stemmer = PorterStemmer()
    for essay in read_all_essays():
        words = word_tokenize(essay)
        stop_words = stopwords.words('english')
        words_cleaned = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words and word not in EXCLUDED_CLEAN_WORDS]
        essay_cleaned = " ".join(words_cleaned)
        cleaned_essays.append(essay_cleaned)
    return cleaned_essays

def spellcheck_essays():
    errors = []
    for essay in read_all_essays():
        words = word_tokenize(essay)
        stop_words = stopwords.words('english')
        words_cleaned = [word for word in words if word.isalpha() and word not in stop_words and word not in EXCLUDED_CLEAN_WORDS]
        errors.append(spellcheck_errors(words_cleaned))
    return np.array(errors)

def spellcheck_errors(words):
    count = 0
    for word in words:
        if word not in WORD_CORPUS:
            count += 1
    return count

def get_avg_sentence_length():
    count = []
    for essay in read_all_essays():
        sentences = sent_tokenize(essay)
        words = word_tokenize(essay)
        count.append(len(words) / len(sentences))

    return np.array(count)


def Tfidf_vectorize_data(data):
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=4, norm="l2")
    X = vectorizer.fit_transform(data)
    return pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

def preprocess_vocabulary_data():
    essays = clean_data()
    df = Tfidf_vectorize_data(essays)
    df["unique_words"] = np.array(list(map(lambda essay: len(set(essay.split(" "))),essays)))
    df["word_count"] = np.array(list(map(lambda essay: len(essay.split(" ")) ,essays)))
    df["avg_sentence_length"] = get_avg_sentence_length()
    df["spelling_errors"] = spellcheck_essays()
    #df["avg_word_length"] = np.array(list(map(lambda essay: sum(list(map(len,essay.split(" ")))) / len(essay.split(" ")) ,essays)))
    return df

def preprocess_structure_data():
    df = pd.DataFrame()

    noun_counts = []
    verb_counts = []
    adjective_counts = []
    adverb_counts = []
    comma_counts = []
    period_counts = []
    noun_cnt = verb_cnt = adjective_cnt = adverb_cnt = comma_cnt = period_cnt = 0

    for essay in read_all_essays():
        essay_tokenized = word_tokenize(essay)
        comma_cnt = essay_tokenized.count(",")
        period_cnt = essay_tokenized.count(".")
        sentence_cnt = len(sent_tokenize(essay))
        pos_tags = nltk.pos_tag(essay_tokenized)

        for word,tag in pos_tags:
            if tag in NOUNS: noun_cnt += 1
            elif tag in ADVERBS: adverb_cnt += 1
            elif tag in ADJECTIVES: adjective_cnt += 1
            elif tag in VERBS: verb_cnt += 1

        noun_counts.append(noun_cnt / sentence_cnt)
        verb_counts.append(verb_cnt / sentence_cnt)
        adjective_counts.append(adjective_cnt / sentence_cnt)
        adverb_counts.append(adverb_cnt / sentence_cnt)
        period_counts.append(period_cnt / len(essay_tokenized))
        comma_counts.append(comma_cnt / sentence_cnt)

        noun_cnt = verb_cnt = adjective_cnt = adverb_cnt = 0

    df["noun_count_avg"] = np.array(noun_counts)
    df["verb_count_avg"] = np.array(verb_counts)
    df["adjective_count_avg"] = np.array(adjective_counts)
    df["adverb_count_avg"] = np.array(adverb_counts)
    df["avg_comma_count_per_sentence"] = np.array(comma_counts)
    # df["avg_period_count"] = np.array(period_counts)
    #df["sentence_count_avg"] = get_sentence_count()
    #df["spelling_errors"] = spellcheck_essays()

    return df


def cross_validate_model(model,X,y):
    scores = cross_val_score(model.classifier, X.values, np.array(y.values).ravel(), cv=5)
    print(np.average(scores))


def test_vocabulary_scoring(vocabulary_scores):
    voc_df = preprocess_vocabulary_data()
    X_train, X_test, y_train, y_test = train_test_split(voc_df, vocabulary_scores, test_size=0.2, random_state=22)
    voc_scorer = EssayVocabularyScorer(X_train, y_train)
    cross_validate_model(voc_scorer, X_test, y_test)

def test_structure_scoring(structure_scores):
    struct_df = preprocess_structure_data()
    X_train, X_test, y_train, y_test = train_test_split(struct_df, structure_scores, test_size = 0.2, random_state = 22)
    struct_scorer = EssayStructureScorer(X_train,y_train)
    cross_validate_model(struct_scorer,X_test,y_test)


if __name__ == "__main__":
    structure_scores = pd.read_csv("data/essays_structure_score.txt")
    vocabulary_scores = pd.read_csv("data/essays_vocabulary_score.txt")

    # test_vocabulary_scoring(vocabulary_scores)
    # test_structure_scoring(structure_scores)
    #find_missing_words()
    word_replacer = EssayMissingWordsReplacer()
    print(word_replacer(read_essay(2)))
# previous best: 0.67718
# with word count = 0.73527