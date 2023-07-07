import tensorflow
import numpy
import numpy
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
#using this series to learn: https://www.youtube.com/watch?v=wypVcNIH6D4&list=PLzMcBGfZo4-ndH9FoC4YWHGXG5RZekt-Q&index=1&t=0s
#note, you have to learn the intent of a message, hence "intents" is meant to give repsonses to what vibe the program believes the conversation is giving off, refer to tutorial
#https://www.techwithtim.net/tutorials/ai-chatbot/chat-bot-part-2

stemmer = LancasterStemmer()
words = [] #
lables = []
docs_patterns = []
docs_tags = []


with open("intents.json") as file:
    data = json.load(file)

for intent in data["intents"]:
    for pattern in intent["patterns"]: #stems the words/takes the root word in the words in the patterns
        wrds = nltk.word_tokenize(pattern) #returns a list of the tokenized words into a list
        words.extend(wrds)
        docs_patterns.append(pattern)
        docs_tags.append(intent["tag"]) #important for classifying the pattern to a specific name of intent

    if intent["tag"] not in lables:
        lables.append(intent["tag"])

#removes any duplicates in the words list
words = [stemmer.stem(w.lower()) for w in words] 
words = sorted(list(set(words)))


lables = sorted(lables)

#since neuro networks only understand numbers, this converts the lists into "bags of words" or the amount of times a word appears in a sentence 

training = []
output = []

out_empty = [0 for _ in range(len(intent["tag"]))]

