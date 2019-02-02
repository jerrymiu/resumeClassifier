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
maxLength = 0

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
	processedTokens = map(lambda token: str(vocabulary[token]), tokens)
	return processedTokens

category = rawData['Category'].tolist()
trainData = rawData['Resume'].tolist()

# Process data
trainData = map(convertWordToIndex, trainData)
trainData = keras.preprocessing.sequence.pad_sequences(trainData, value=0, padding='post', maxlen=3952)

lenVocab = len(vocabulary)
vocabulary["<PAD>"] = 0