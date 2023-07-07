import tensorflow
import numpy
import numpy as np
import random
import tflearn
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

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
        docs_patterns.append(wrds)
        docs_tags.append(intent["tag"]) #important for classifying the pattern to a specific name of intent

    if intent["tag"] not in lables:
        lables.append(intent["tag"])

#removes any duplicates in the words list
words = [stemmer.stem(w.lower()) for w in words if w != "?"] 
words = sorted(list(set(words)))


lables = sorted(lables)

#since neuro networks only understand numbers, this converts the lists into "bags of words" or the amount of times a word appears in a sentence 

training = []
output = []

#turns it into a bag of words
out_empty = [0 for _ in range(len(lables))]

for index, doc in enumerate(docs_patterns):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    ourput_row = out_empty[:]
    ourput_row[lables.index(docs_tags[index])] = 1

    training.append(bag)
    output.append(ourput_row)

training = np.array(training)
ourput = np.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) #first hidden layer that has 8 nuerons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #gives us the probability of each neuron in this layer and that will be the output layer
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #epoch change the number to test it
model.save("model.tflearn")