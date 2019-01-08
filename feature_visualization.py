import pandas as pd
import numpy as np
import nltk
import operator
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from utils import read_all_essays
from visualization import wordcloud


#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
VISUALIZE_N_WORDS = 12


def sentences_count():
    count = []
    for essay in read_all_essays():
        count.append(len(sent_tokenize(essay)))
    return count

def words_count():
    count =[]
    for essay in read_all_essays():
        count.append(len(word_tokenize(essay)))
    return count


def count_word_occurences():
    corpus = dict()
    # lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for essay in read_all_essays():
        words = word_tokenize(essay)
        stop_words = stopwords.words('english')
        words_cleaned = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
        for word in words_cleaned:
            if word not in corpus.keys():
                corpus[word] = 1
            else:
                corpus[word] += 1
    return corpus

def common_words_graph():
    corpus = count_word_occurences()
    sorted_corpus = sorted(corpus.items(),key=lambda kv: kv[1],reverse=True)
    labels = [k for k, v in sorted_corpus[1:VISUALIZE_N_WORDS]]
    values = [v for k, v in sorted_corpus[1:VISUALIZE_N_WORDS]]
    plt.bar(np.arange(len(values)),values, color="green")
    plt.xticks(range(len(labels)), list(labels))
    # plt.plot(values, '-', color="green", label="x")
    plt.show()

def show_boxplots():
    word_counts = words_count()
    sentence_counts = sentences_count()
    from visualization import boxplot
    boxplot(["Words"], word_counts, y_label="Number of words", title="Boxplot of word count in essays")
    boxplot(["Sentences"], sentence_counts, y_label="Number of sentences",
            title="Boxplot of sentence count in essays")

def show_wordcloud():
    words = count_word_occurences()
    all_words = sum(words.values())
    for key in words.keys():
        words[key] = float(words[key]) / all_words

    wordcloud(words,"")


#common_words_graph()
#show_boxplots()
show_wordcloud()