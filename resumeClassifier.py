# Importing TensorFlow and keras
import tensorflow as tf
from tensorflow import keras

# Helpers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rawData = pd.read_csv('resume_dataset.csv')

# Process data
for index, row in rawData.iterrows():
	# Tokenize the text of each resume
	tokens = row[2].split()

	# Remove escape sequences
	tokens = [''.join(tokenChar for tokenChar in rawToken if rawToken.isalnum()) for rawToken in tokens]
	
	# Remove empty tokens
	tokens = [token for token in tokens if token]
	
	# Convert to lowercase

	# Convert each unique word to a dictionary
	

