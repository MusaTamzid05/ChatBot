from nltk.stem import WordNetLemmatizer
import random
import numpy as np


class Trainer:

    def __init__(self , words , classes , documents):
        self.words = words
        self.classes = classes
        self.documents = documents
        self._prepare_data()

    def _prepare_data(self):

        lemmatizer = WordNetLemmatizer()
        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            for word in self.words:
                bag.append(1) if word in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag , output_row])

        random.shuffle(training)
        training = np.array(training)

        self.train_x = list(training[: , 0])
        self.train_y = list(training[: , 1])

        print("[*] Training data created")







