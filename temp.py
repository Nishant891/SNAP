import nltk
from nltk.stem.lancaster import LancasterStemmer
Stemmer = LancasterStemmer()

import numpy
import tflearn
import json
import tensorflow
import random
import pickle
import speech_recognition as sr
import pyttsx3


#Openning the intents.json file and assigning it to a variable, data.

with open("intents.json") as file:
    data = json.load(file)
    
words  =  []
docs_x =  []
docs_y =  []
labels =  []

#try to open the previosly loaded data through a pickle file if not exists the create and load one.

try:
    with open("xyz.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            
            # Tokenizing and appending all the values of list patterns into word
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            #words = ["Hi","How","are","you","Is","anyone"....]
            
            
            docs_x.append(wrds)
            #docs_x = [["Hi"],["How","are","you"],["Is","anyone","there"]...]
            
            docs_y.append(intent["tag"])
            
            if intent["tag"] not in labels:
                    labels.append(intent["tag"])
            #labels = ["greeting","greeting","greeting"...."]
                    
    words = [Stemmer.stem(w) for w in words if w not in "?"]
    words = sorted(list(set(words)))
            
    labels = sorted(labels)
            
    training = []
    output = []
            
    out_empty = [0 for _ in range(len(labels))]
            
    for x,doc in enumerate(docs_x):
            
        bag = []
                
        wrds = [Stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
                
        training.append(bag)
        output.append(output_row)
            
        
    training = numpy.array(training)
    output = numpy.array(output)
    
    #if xyz.pickle does not exists then make a tuple of words,labels,training,output and dump it in yhe pickle file
    with open("xyz.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)


tensorflow.compat.v1.reset_default_graph()
net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")


def bag_of_words(s,words):
    
    bag = [0 for _ in range(len(words))]
    
    s_word = nltk.word_tokenize(s)
    s_word = [Stemmer.stem(w) for w in s_word if w not in "?"]
    
    for x in s_word:
        for i,y in enumerate(words):
            if x == y:
                bag[i] = 1
                
    return numpy.array(bag)


#This function takes input from the user as speech and converts it into text.
def take_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            print("Sorry could not recognize your voice")
            chat()
            
#Initializing chat functions let's us communicate with the bot.
def chat():
    print("Hey there! I am SNAP. How can I help you ?\n")
    
    while True:
        inp = take_input()
        print(inp)
        if inp in ["quit","I am done","Quit","Done","Thank you"]:
            break
        else:
            result = model.predict([bag_of_words(inp,words)])
            print(result)
            print(labels)
            if(result.max() < 0.2):
                print("Sorry I don't know about that.")
            else:
                result_index = numpy.argmax(result)
                print(result_index)
                tag = labels[result_index]
                for tg in data["intents"]:
                    if tg["tag"] == tag:
                        responses = tg["responses"]
                        break
                    
                #The code below converts the text output to speech.
                text_speech = pyttsx3.init()
                text_speech.say(random.choice(responses))
                text_speech.runAndWait()
                    
chat()
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
