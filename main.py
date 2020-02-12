from chatbot.bot import Chatbot

def main():
    chatbot = Chatbot(intents_path = "./intents.json")
    chatbot.train(save_path = "chatbot_train1")

if __name__ == "__main__":
    main()

