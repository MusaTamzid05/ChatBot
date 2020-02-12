

class Chatbot:

    def __init__(self , intents_path):
        self.intents_path = intents_path
        self.train_model_loaded = False

    def train(self , save_path):
        pass

    def load_train_model(self , path):
        pass


    def response(self , sentence):

        if self.train_model_loaded == False:
            print["[-] Please load the train model by calling load_train_model or train from scratch by calling train first"]
            return
