import keras
import tensorflow as tf

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    

if __name__ == "__main__":
    #test materials
    x = tf.ones((2, 2))
    linear_layer = Linear(4, 2)
    y = linear_layer(x)
    print(y)