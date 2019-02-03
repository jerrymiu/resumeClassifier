# Importing TensorFlow and keras
import tensorflow as tf
from tensorflow import keras

# Helpers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords

from collections import defaultdict
from itertools import count

rawData = pd.read_csv('resume_dataset.csv')
stopWords = set(stopwords.words('english'))
vocabulary = defaultdict(count(1).next)
categoryNames = defaultdict(count(0).next)

def convertWordToIndex(resumeText):
	# Tokenize the text of each resume
	tokens = resumeText.split()

	# Remove escape sequences
	tokens = [''.join(tokenChar for tokenChar in rawToken if rawToken.isalnum()) for rawToken in tokens]
	
	# Remove empty tokens, remove stopwords and convert to lowercase
	tokens = [token.lower() for token in tokens if token and token not in stopWords]
	
	# Store each unique word to the vocabulary dictionary
	for token in tokens:
		vocabulary[token]
	
	# Convert the tokens for each resume into a list of integers
	processedTokens = map(lambda token: vocabulary[token], tokens)
	return processedTokens

trainLabels = rawData['Category'].tolist()
trainData = rawData['Resume'].tolist()

# Process Labels
for trainLabel in trainLabels:
	categoryNames[trainLabel]

trainLabels = map(lambda trainLabel: categoryNames[trainLabel], trainLabels)

# Process data
trainData = map(convertWordToIndex, trainData)
trainData = keras.preprocessing.sequence.pad_sequences(trainData, value=0, padding='post', maxlen=3952)

lenVocab = len(vocabulary)
vocabulary["<PAD>"] = 0

model = keras.Sequential()
model.add(keras.layers.Embedding(lenVocab+1, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(300, activation=tf.nn.relu))
model.add(keras.layers.Dense(25, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

npTrainData = np.asarray(trainData, dtype=np.int32)
npTrainLabels = np.asarray(trainLabels, dtype=np.int32)

history = model.fit(npTrainData, npTrainLabels, epochs=5)

results = model.evaluate(npTrainData, npTrainLabels)
