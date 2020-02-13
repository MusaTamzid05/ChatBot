from keras.models import load_model
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import random
import os


class Predictor:

    def __init__(self , intents ,  model_path):
        self.model = load_model(os.path.join(model_path , "model.h5"))

        self.intents = intents
        self.words = pickle.load(open(os.path.join(model_path , "words.pkl") , "rb"))
        self.classes = pickle.load(open(os.path.join(model_path , "classes.pkl"), "rb"))

        self.lemmatizer = WordNetLemmatizer()

    def clean_up(self , sentence):


        words = nltk.word_tokenize(sentence)
        words = [self.lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bow(self , sentence):

        sentence_words = self.clean_up(sentence)
        bag = [0] * len(self.words)

        for sentence_word in sentence_words:
            for i , word in enumerate(self.words):
                if word == sentence_word:
                    bag[i] = 1

        return np.array(bag)

    def predict(self , sentence , error_threshold = 0.25):

        bow_data = self.bow(sentence)
        res = self.model.predict(np.array([bow_data]))[0]
        results = [[i , r] for i , r in enumerate(res) if r > error_threshold]
        results.sort(key = lambda x : x[1] , reverse = True)
        return_list = []

        for result in results:
            return_list.append({"intent" : self.classes[result[0]] , "probability" : result[1]})

        return return_list

    def get_response_to(self , sentence):

        responses = self.predict(sentence)
        tag = responses[0]["intent"]

        intents = self.intents["intents"]
        result = None

        for intent in intents:
            if intent["tag"] == tag:
                result = random.choice(intent["responses"])
                break

        return result







def main():

    predictor = Predictor(path = "./chatbot_model.h5")

    while True:
        print(predictor.get_response_to(input(">>")))



if __name__ == "__main__":
    main()




