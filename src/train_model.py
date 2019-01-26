from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import pickle

# TODO: NER filtern damit keine Namen im Text enthalten sind.
import spacy
# nlp = spacy.load("en")
# for token in nlp(text).ents:
#     print(token.)


from sklearn.feature_extraction.text import CountVectorizer

nltk.download("punkt")

"""
Constants
"""
SQUENCE_LENGTH = 40
MAX_FEATURES = 1000

"""
load the data from our dataset
"""
with open("./data/jane_austen.txt", "r", encoding="utf-8") as file:
    contents = file.read()
data = "\n".join(contents.split("\n"))


"""
split the whole text into word tokens related to a CountVectorizer
"""
tokens = word_tokenize(data)

print("Tokens: ", len(tokens))
print("Set of Tokens: ", len(set(tokens)))

cv = CountVectorizer(lowercase=False, token_pattern="(.*)", max_features=MAX_FEATURES)

# Hiermit wird ein Vokabular aufgebaut, welches sich über den gesamten input Text streckt
cv.fit(tokens)

# Dies muss nun extrahiert werden // Array mapping from feature integer indices to feature name
# gibt eine liste mit allen wörtern zürück.
features = cv.get_feature_names()


"""
specify a number for each word since the neural network can only handle numbers
"""
word_to_int = {}
int_to_word = {}

# get the index for each word in a dictionary
for i in range(0, len(features)):
    word = tokens[i]
    word_to_int[word] = i
    int_to_word[i] = word

#convert each word to the index in the word_to_int dictionary
tokens_transformed = [word_to_int[word] for word in tokens if word in word_to_int]


"""
Preparation for training
"""
# MARK: - One Hot Encoding it not required since the Embedding Layer will do this
X = []
y = []


for i in range(0, len(tokens_transformed) - SQUENCE_LENGTH):
    X.append(tokens_transformed[i:i + SQUENCE_LENGTH])
    y.append(tokens_transformed[i + SQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print("X Shape: ", X.shape)
print("y Shape: ", y.shape)


"""
Build the model
"""
model = Sequential()
model.add(Embedding(cv.max_features, 100, input_shape=(SQUENCE_LENGTH,)))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(cv.max_features, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

"""
Train the neural network
"""
checkpoint = ModelCheckpoint("./models/weights/jane_austen.{epoch:02d}-{val_loss:.2f}.hdf5")
model.fit(
    X,
    to_categorical(y, num_classes=cv.max_features),
    epochs=1,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint]
)


"""
Get a summery and save the data
"""
model.summary()
model.save("./models/jane_austen.model")

with open("./pickles/word_to_int.pickle", "wb") as file:
    pickle.dump(word_to_int, file)

with open("./pickles/int_to_word.pickle", "wb") as file:
    pickle.dump(int_to_word, file)