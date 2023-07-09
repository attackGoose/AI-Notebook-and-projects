import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, "rb").read().decode(encoding="utf-8")

vocab = sorted(set(text))

#print(f"Length of test: {len(text)} characters")

#changing the letters to a numerical representation before testing:
sample_characters = ['abcdefghijklmnopqrstuvw', 'xyz']

#creates tokens from the list of letters
chars = tf.strings.unicode_split(sample_characters, input_encoding='UTF-8')

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

chars = ids_from_chars(chars)


chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

tf.strings.reduce_join(chars, axis=-1).numpy()


def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))