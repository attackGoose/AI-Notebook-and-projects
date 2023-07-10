#https://youtu.be/tpCFfeUEGs8

import tensorflow as tf

print(tf.__version__)

scalar = tf.constant(10)

unchangeable_tensor_vector = tf.constant([10,10])

changeable_tensor_vector = tf.Variable([10,7])

changeable_tensor_vector[0].assign(7)
print(changeable_tensor_vector)

