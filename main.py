import tensorflow
import numpy
import numpy
import random
import tflearn
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

#using this series to learn: https://www.youtube.com/watch?v=wypVcNIH6D4&list=PLzMcBGfZo4-ndH9FoC4YWHGXG5RZekt-Q&index=1&t=0s
#note, you have to learn the intent of a message, hence "intents" is meant to give repsonses to what vibe the program believes the conversation is giving off, refer to tutorial
#https://www.techwithtim.net/tutorials/ai-chatbot/chat-bot-part-2

stemmer = LancasterStemmer()
words = []
lables = []
docs_x = []
docs_y = []


with open("intents.json") as file:
    data = json.load(file)

for intent in data["intents"]:
    for pattern in intent["patterns"]: #stems the words/takes the root word in the words in the patterns
        wrds = nltk.word_tokenizer(pattern) #returns a list of the tokenized words into a list
        words.extend(wrds)
        docs_x.append(pattern)

    if intent["tag"] not in lables:
        lables.append(intent["tag"])

    