import json
from chatbot.preprocessor import Preprocessor
from chatbot.trainer import Trainer
from chatbot.predictor import Predictor

class Chatbot:

    def __init__(self , intents_path):
        self._load_intents_from(intents_path)
        self.train_model_loaded = False

    def _load_intents_from(self , path):
        data_file = open(path).read()
        self.intents = json.loads(data_file)

    def train(self , save_path):
        preprocessor = Preprocessor(self.intents)
        preprocessor.run(save_path)

        trainer = Trainer(preprocessor.words , preprocessor.classes , preprocessor.documents)
        trainer.train(save_path)

    def load_train_model(self , path):

        self.predictor = Predictor(self.intents , path)
        self.train_model_loaded = True



    def response(self , sentence):

        if self.train_model_loaded == False:
            print["[-] Please load the train model by calling load_train_model or train from scratch by calling train first"]
            return

        return self.predictor.get_response_to(sentence)
