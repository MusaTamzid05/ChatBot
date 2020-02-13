from chatbot.bot import Chatbot

def train_example():
    chatbot = Chatbot(intents_path = "./intents.json")
    chatbot.train(save_path = "chatbot_train1")


def response_example():
    chatbot = Chatbot(intents_path = "./intents.json")
    chatbot.load_train_model("./chatbot_train1")

    while True:
        print(chatbot.response(input(">>> ")))

if __name__ == "__main__":
    response_example()

