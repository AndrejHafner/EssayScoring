import random

import numpy as np
import re

from nltk import word_tokenize
from sklearn.svm import SVC
from nltk.corpus import stopwords
from utils import read_all_essays, read_books, read_essay, random_date, random_time
from nltk.tag.stanford import StanfordNERTagger
from random import randint

import os
java_path = "C:/Program Files (x86)/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

# st = StanfordNERTagger('stanford_ner/english.muc.7class.distsim.crf.ser.gz', 'stanford_ner/stanford-ner.jar',encoding='utf-8')

MISSING_WORD_TYPES = ['num','caps','person','month','date','location','organization','time','percent','money']

DEFAULT_REPLACEMENTS = {'person': ["Andrew", "Nick","Michael","Jane","Hillary"],
                        'month': ['January','February','June','August','September'],
                        'location': ['America','Europe','Kentucky','town','school'],
                        'organization':['hospital','JPMorgan','town hall','DMW','prison']}

stop_words = stopwords.words('english')

class EssayVocabularyScorer(object):

    def __init__(self,train_df,Y):
        self.train_df = train_df
        self.Y = Y
        self.train_model()

    def __call__(self,X):
        return self.classifier.predict(X)

    def train_model(self):
        self.classifier = SVC()
        self.classifier.fit(self.train_df.values, np.array(self.Y.values).ravel())



class EssayStructureScorer(object):

    def __init__(self,train_df,Y):
        self.train_df = train_df
        self.Y = Y
        self.train_model()

    def __call__(self,X):
        return self.classifier.predict(X)

    def train_model(self):
        self.classifier = SVC()
        self.classifier.fit(self.train_df.values, np.array(self.Y.values).ravel())

class EssayMissingWordsReplacer(object):

    def __init__(self):
        self.train_model()

    def __call__(self, text):
        return self.replace_missing_words(text)

    def replace_missing_words(self,text):
        essay_tokenized = re.split(r'[\s\n]+',text)
        essay_tokenized_cleaned = [word.replace("&gt;", " ").replace("&lt;", " ").replace(".", " ").strip() for word in essay_tokenized]
        missing_words_replacement = {}
        for i in range(0,len(essay_tokenized_cleaned) - 2):
            word = essay_tokenized_cleaned[i + 2]
            if word.startswith("@") and word not in missing_words_replacement.keys():
                before_1 = essay_tokenized_cleaned[i + 1]
                before_2 = essay_tokenized_cleaned[i]
                word = essay_tokenized_cleaned[i + 2]
                word_type = [wtype for wtype in MISSING_WORD_TYPES if wtype in word]

                missing_words_replacement[word] = self.find_replacement(before_1,before_2,word_type[0])
        return missing_words_replacement


    def find_replacement(self,before_1,before_2,word_type):
        if before_1.startswith('@') or before_2.startswith('@'): return
        before_1_words = self.before_words_freq["before_1"][before_1]
        before_2_words = self.before_words_freq["before_2"][before_2]

        contained_in_both_set = set(before_1_words.keys()) & set(before_2_words.keys())
        # e = st.tag(word_tokenize("America"))
        # test = [(word,st.tag([word])) for word in contained_in_both_set]

        same_word_freq_sum = [(word,before_1_words[word] + before_2_words[word] * 0.5) for word in contained_in_both_set if word not in stop_words]
        same_word_freq_sum = sorted(same_word_freq_sum,key=lambda item: item[1],reverse=True)
        #top_ner_tags = [st.tag([word]) for word,score in same_word_freq_sum[:3]]

        #filtered = filter(lambda el:el[1] == word_type,top_ner_tags)
        if len(same_word_freq_sum) == 0:
            return self.get_default_words(word_type)

        return same_word_freq_sum[0][0]

    def get_default_words(self,type):
        if type in DEFAULT_REPLACEMENTS.keys():
            return DEFAULT_REPLACEMENTS[type][randint(0,4)]

        if type == 'num':
            return randint(1,100)
        elif type == 'caps':
            return "WOW"
        elif type == 'date':
            return random_date("1/1/20017 1:30 PM", "1/1/2018 4:50 AM",random.random())
        elif type == 'time':
            return random_time("1/1/20017 1:30 PM", "1/1/2018 4:50 AM", random.random())
        elif type == 'percent':
            return f"{randint(0,100)}%"
        elif type == 'money':
            return f"{randint(0,1000)}€"
        else:
            return ''

    def train_model(self):
        # "before_1" is the first word before the word being predicted and "before_2" the second one
        # "before_2" "before_1" missing_word "after_1" "after_2"
        self.before_words_freq = {"before_1":{}, "before_2":{}}

        for essay in read_all_essays():
            essay_tokenized = self.tokenize_text(essay)
            self.before_word_frequency(essay_tokenized)


        for book in read_books():
            book_tokenized = self.tokenize_text(book)
            self.before_word_frequency(book_tokenized)

    def tokenize_text(self,essay):
        essay_tokenized = re.split(r'[\s\n]+', essay)
        essay_tokenized = [word.replace("&gt;", "").replace("&lt;", "").replace(".", "").strip() for word in essay_tokenized if not word.startswith('@') and not word.startswith('"@')]
        return essay_tokenized

    def before_word_frequency(self, text):
        for i in range(0, len(text) - 2):
            # if text[i].startswith('@'):
            #     self.missing_set.add(re.sub('[\W\d_]', '',text[i][1:]))
            if text[i + 1] in self.before_words_freq["before_1"].keys():
                if text[i + 2] in self.before_words_freq["before_1"][text[i + 1]].keys():
                    self.before_words_freq["before_1"][text[i + 1]][text[i + 2]] += 1
                else:
                    self.before_words_freq["before_1"][text[i + 1]][text[i + 2]] = 1
            else:
                self.before_words_freq["before_1"][text[i + 1]] = {}
                self.before_words_freq["before_1"][text[i + 1]][text[i + 2]] = 1

            if text[i] in self.before_words_freq["before_2"].keys():
                if text[i + 2] in self.before_words_freq["before_2"][text[i]].keys():
                    self.before_words_freq["before_2"][text[i]][text[i + 2]] += 1
                else:
                    self.before_words_freq["before_2"][text[i]][text[i + 2]] = 1
            else:
                self.before_words_freq["before_2"][text[i]] = {}
                self.before_words_freq["before_2"][text[i]][text[i + 2]] = 1

 #
 # def before_word_frequency(self, text):
 #        for i in range(0, len(text) - 2):
 #            if text[i + 2] in self.before_words_freq["before_1"].keys():
 #                if text[i + 1] in self.before_words_freq["before_1"][text[i + 2]].keys():
 #                    self.before_words_freq["before_1"][text[i + 2]][text[i + 1]] += 1
 #                else:
 #                    self.before_words_freq["before_1"][text[i + 2]][text[i + 1]] = 1
 #            else:
 #                self.before_words_freq["before_1"][text[i + 2]] = {}
 #                self.before_words_freq["before_1"][text[i + 2]][text[i + 1]] = 1
 #
 #            if text[i + 2] in self.before_words_freq["before_2"].keys():
 #                if text[i] in self.before_words_freq["before_2"][text[i + 2]].keys():
 #                    self.before_words_freq["before_2"][text[i + 2]][text[i]] += 1
 #                else:
 #                    self.before_words_freq["before_2"][text[i + 2]][text[i]] = 1
 #            else:
 #                self.before_words_freq["before_2"][text[i + 2]] = {}
 #                self.before_words_freq["before_2"][text[i + 2]][text[i]] = 1