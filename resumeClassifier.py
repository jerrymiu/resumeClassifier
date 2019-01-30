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
vocabulary = defaultdict(count(0).next)
# Process data
for index, row in rawData.iterrows():
	# Tokenize the text of each resume
	tokens = row[2].split()

	# Remove escape sequences
	tokens = [''.join(tokenChar for tokenChar in rawToken if rawToken.isalnum()) for rawToken in tokens]
	
	# Remove empty tokens, remove stopwords and convert to lowercase
	tokens = [token.lower() for token in tokens if token and token not in stopWords]
	
	# Convert each unique word to a dictionary
	for token in tokens:
		vocabulary[token]

