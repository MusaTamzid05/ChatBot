from nltk.stem import WordNetLemmatizer
import random
import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD

import os
from matplotlib import pyplot as plt


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

    def _init_model(self , learning_date):

        model = Sequential()
        model.add(Dense(128 , input_shape = (len(self.train_x[0]) , ) , activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64,  activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y[0]) , activation = "softmax"))
        sgd = SGD(lr = learning_date , decay = 1e-6 , momentum = 0.9 , nesterov = True)
        model.compile(loss = "categorical_crossentropy" , optimizer = sgd , metrics = ["accuracy"])

        self.model = model
        self.model.summary()


    def train(self , save_path ,  epochs = 200 , batch_size = 5 , learning_date = 0.01):
        self._init_model(learning_date)

        history = self.model.fit(np.array(self.train_x) , np.array(self.train_y) ,
                                epochs = epochs ,
                                batch_size = batch_size ,
                                verbose = 1)

        path = os.path.join(save_path , "model.h5")
        self.model.save(path)
        print("Model saved in {}".format(path))
        self.train_model_loaded = True
        self.visualize(history.history)


    def visualize(self , history):

        acc = history["acc"]
        loss = history["loss"]
        epochs = range(1 , len(acc) + 1)

        plt.plot(epochs , acc , "bo" , label = "Trainning acc")
        plt.plot(epochs , loss , "ro" , label = "Trainning loss ")
        plt.legend()
        plt.show()


