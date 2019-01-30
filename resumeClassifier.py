# Importing TensorFlow and keras
import tensorflow as tf
from tensorflow import keras

# Helpers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rawData = pd.read_csv('resume_dataset.csv')
print(rawData)

#Convert words to an integer

