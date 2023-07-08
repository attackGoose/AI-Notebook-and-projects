#make and train model in this file

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

#applying a RNN (recurrent Neuro network)

batch_size = 64
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28

units = 64
output_size = 10  # labels are from 0 to 9

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
sample, sample_lable = x_train[0], y_train[0]

model = build_model(allow_cudnn_kernel=True)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1
)





#https://www.tensorflow.org/guide/keras/working_with_rnns




#class Linear(keras.layers.Layer):
#    def __init__(self, units=32, input_dim=32):
#        super().__init__()
#        self.w = self.add_weight(
#            shape=(input_dim, units), initializer="random_normal", trainable=True
#        )
#        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)
#
#    def call(self, inputs):
#        return tf.matmul(inputs, self.w) + self.b
    

#if __name__ == "__main__":
#    #test materials
#    x = tf.ones((2, 2))
#    linear_layer = Linear(4, 2)
#    y = linear_layer(x)
#    print(y)

