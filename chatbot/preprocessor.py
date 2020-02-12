import nltk

class Preprocessor:

    def __init__(self , intents):

        self.intents = intents
        self.words = []
        self.classes = []
        self.documents = []

    def _load_data(self):

        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                current_words = nltk.word_tokenize(pattern)
                self.words += current_words
                self.documents.append((current_words , intent["tag"]))

                if intent["tag"] not in self.classes:
                    self.classes.append(intent["tag"])


        print(self.words)
        print(self.documents)
        print(self.classes)

    def process(self):
        self._load_data()


