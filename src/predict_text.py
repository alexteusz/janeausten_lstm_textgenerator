import nltk
import pickle
from keras.models import load_model
import numpy as np

with open("../data/jane_austen.txt", "r", encoding="utf-8") as file:
    contents = file.read()

contents = contents.split("\n")
contents = [line.strip() for line in contents]

contents = "\n".join(contents)

#contents = contents.replace("\n", " \\n ")


#nltk.download('punkt')

tokens = nltk.word_tokenize(contents)


with open("../results/1000Words_10Epochs/pickles/word_to_int.pickle", "rb") as file:
    word_to_int = pickle.load(file)

with open("../results/1000Words_10Epochs/pickles/int_to_word.pickle", "rb") as file:
    int_to_word = pickle.load(file)

tokens_transformed = [word_to_int[word] for word in tokens if word in word_to_int]


"""
load model 
"""
model = load_model("../results/1000Words_10Epochs/jane_austen.model")

""""
predict text with random parts  
"""

sentence = np.array(tokens_transformed[1000:1040])


for i in range(0, 100):
    prediction = model.predict(sentence.reshape(1, 40))

    #word = np.argmax(prediction[0])
    word = np.random.choice(len(int_to_word), p=prediction[0])

    print(int_to_word[word].replace(",","").replace(".",""), end=" ")

    sentence = np.append(sentence[1:], [word])