import numpy as np

from sklearn.svm import SVC



class EssayVocabularyScorer(object):

    def __init__(self,train_df,Y):
        self.train_df = train_df
        self.Y = Y
        self.train_model()

    def __call__(self,X):
        return self.classifier.predict(X)

    def train_model(self):
        self.classifier = SVC()
        self.classifier.fit(self.train_df.values, np.array(self.Y.values))



class EssayStructureScorer(object):

    def __init__(self):
        pass

    def __call__(self):
        pass
