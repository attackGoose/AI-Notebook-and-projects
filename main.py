import tensorflow as tf
from tensorflow import keras
import numpy
import numpy as np
import random
import json
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer

#https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
#https://www.geeksforgeeks.org/deploy-a-chatbot-using-tensorflow-in-python/
#using this series to learn: https://www.youtube.com/watch?v=wypVcNIH6D4&list=PLzMcBGfZo4-ndH9FoC4YWHGXG5RZekt-Q&index=1&t=0s
#note, you have to learn the intent of a message, hence "intents" is meant to give repsonses to what vibe the program believes the conversation is giving off, refer to tutorial
#https://www.techwithtim.net/tutorials/ai-chatbot/chat-bot-part-2




with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, lables, training, output = pickle.load(f)
except:
    stemmer = LancasterStemmer()
    words = [] #
    lables = []
    docs_patterns = []
    docs_tags = []

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

    with open("data.pickle", "wb") as f:
        pickle.dump((words, lables, training, output), f)

#creates the neuro network and trains it/loads its training
tf.reset_default_graph()

#NOTE:replace this with keras's neuro network and training



#net = tflearn.input_data(shape=[None, len(training[0])])
#net = tflearn.fully_connected(net, 8) #first hidden layer that has 8 nuerons
#net = tflearn.fully_connected(net, 8)
#net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #gives us the probability of each neuron in this layer and that will be the output layer
#net = tflearn.regression(net)

#model = tflearn.DNN(net)

try:
    #model.load("model.tflearn")
    pass
except:
    #model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #epoch change the number to test it
    #model.save("model.tflearn")
    pass


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.tokenize(s)
    s_words = [stemmer.stem(words.lower()) for word in s_words]

    for se in s_words:
        for index, wrd in enumerate(words):
            if w == se:
                bag[index] = 1
    
    return numpy.array(bag)

if __name__ == "__main__":
    print("starting chat (type quit to stop)")
    while True:
        text = input("You: ")
        if text.lower() == "quit":
            break

        results = model.predict([bag_of_words(text, words)]) #this shows the accuracy/probability of what the bot thinks is the right response
        results_index = numpy.argmax(results) #chooses the greatest possibility of the correct reply
        tag = lables[results_index]


        if results[results_index] > 0.7: #if the response is above a certain probability it'll print it, otherwise it'll return a confused statement
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responces"]
        else:
            for tg in data["intents"]:
                if tg["tag"] == "confused":
                    responses = tg["responces"]
        
        print(random.choice(responses))

            

