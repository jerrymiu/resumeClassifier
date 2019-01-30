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

def getWordFromIndex(wordIndex):
	if isinstance(wordIndex, str):
		wordIndex = int(wordIndex)
	for word, index in vocabulary.items():
		if wordIndex == index:
			return word
	return "Not Found"

# Process data
for index, row in rawData.iterrows():
	# Tokenize the text of each resume
	tokens = row[2].split()

	# Remove escape sequences
	tokens = [''.join(tokenChar for tokenChar in rawToken if rawToken.isalnum()) for rawToken in tokens]
	
	# Remove empty tokens, remove stopwords and convert to lowercase
	tokens = [token.lower() for token in tokens if token and token not in stopWords]
	maxLength = max(len(tokens), maxLength)
	# Store each unique word to the vocabulary dictionary
	for token in tokens:
		vocabulary[token]
	
	# Convert the tokens for each resume into a list of integers
	processedTokens = map(lambda token: str(vocabulary[token]), tokens)
	rawData.loc[index, 2] = ' '.join(processedTokens)

deciphered = map(getWordFromIndex, rawData.loc[1, 2].split())

lenVocab = len(vocabulary)
vocabulary["<PAD>"] = 0

