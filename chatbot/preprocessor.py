import nltk
from nltk.stem import WordNetLemmatizer

import pickle
import os

class Preprocessor:

    def __init__(self , intents ,  verbose = False):

        self.intents = intents
        self.verbose = verbose
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ["!" , "?"]

    def _load_data(self):

        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                current_words = nltk.word_tokenize(pattern)
                self.words += current_words
                self.documents.append((current_words , intent["tag"]))

                if intent["tag"] not in self.classes:
                    self.classes.append(intent["tag"])

        if self.verbose:
            print("Words : {}".format(self.words))
            print("Documents : {}".format(self.documents))
            print("Clasess: {}".format(self.classes))

    def _process(self):

        lemmatizer = WordNetLemmatizer()

        self.words = [lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_words]

        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        print("Total words : {}".format(len(self.words)))
        print("Total documents : {}".format(len(self.documents)))
        print("Total classes : {}".format(len(self.classes)))


    def run(self, save_path):
        self._load_data()
        self._process()

        if os.path.isdir(save_path):
            raise RuntimeError("{} already exists.".format(save_path))

        os.makedirs(save_path)

        pickle.dump(self.words , open(os.path.join(save_path , "words.pkl" ) , "wb"))
        pickle.dump(self.classes, open(os.path.join(save_path , "classes.pkl" ) , "wb"))
        pickle.dump(self.documents, open(os.path.join(save_path , "documents.pkl" ) , "wb"))

        return self.words , self.classes , self.documents





