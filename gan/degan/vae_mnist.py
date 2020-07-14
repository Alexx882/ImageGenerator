import vae
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def load_prepared_data():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    return (x_train, x_test)


def show_sample(sample):
    plt.imshow(sample)
    plt.colorbar()
    plt.show()


def visualize(vae):
    (_, x_test) = load_prepared_data()

    decoded_imgs = vae.autoencoder.predict(x_test)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def train(vae):
    (x_train, x_test) = load_prepared_data()
    vae.train(x_train, x_test, 50, 256)
    vae.export("gan/degan/vae")


def test():
    (x_train, _) = load_prepared_data()

    show_sample(x_train[0].reshape(28, 28))


if __name__ == "__main__":
    # map the MNIST images (28x28 with only 1 channel) to a space of 32 floats
    vae = vae.VAE((28, 28, 1), 128, summary=True)
    vae.import_models("gan/degan/vae")

    # train(vae)
    visualize(vae)
