import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import gan

def show_sample(sample):
    plt.imshow(sample)
    plt.colorbar()
    plt.show()

def rescale_sample(sample):
    sample_rescaled = []

    for row in sample:
        row_prime = []

        for color in row:
            row_prime.append(color / 255.)

        sample_rescaled.append(row_prime)
    
    return sample_rescaled

if __name__ == "__main__":
    gan = gan.GAN((28,28,1), epochs=10)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # bring x_train into 4d shape (id, row, col, channel) and normalize values to [0;1]
    x_train = [np.array(sample) for sample in x_train]
    x_train = np.array([sample.reshape(28,28,1) for sample in x_train]) / 255.
    
    gan.set_training_data(x_train, y_train)
    gan.train()
    img = gan.generate()

    show_sample(img.reshape(28,28))