import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class MNIST_provider:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def get_numbers(self):
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        # merge, normalize
        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

        np.random.shuffle(mnist_digits)

        current_batch = [] # the current batch to fill

        for sample in mnist_digits:
            current_batch.append(sample)

            if len(current_batch) == self.batch_size:
                yield np.array(current_batch)
                current_batch = []
        
        if len(current_batch) > 0:
            yield np.array(current_batch)
            
if __name__ == "__main__":
    provider = MNIST_provider(128)

    print(provider.get_numbers())