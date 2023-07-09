import tensorflow as tf

import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, "rb").read().decode(encoding="utf-8")

vocab = sorted(set(text))
print(f"{len(vocab)} unique characters")

#print(f"Length of test: {len(text)} characters")

#changing the letters to a numerical representation before testing:
example_texts = ['abcdefg', 'xyz']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

