import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path
import matplotlib.pyplot as plt

ENC_FILENAME = "/encoder.h5"
DEC_FILENAME = "/decoder.h5"
AUT_FILENAME = "/autoencoder.h5"

class VAE:
    E_LAST_LAYER = "latent_layer"

    D_FIRST_LAYER = "decoder"
    D_LAST_LAYER = "decoder_reshape"

    def build_encoder(self):
        self.encoder = keras.Sequential([
            keras.layers.Flatten(input_shape =self.shape),
            keras.layers.Dense(800),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(400),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(200),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(100),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.latent_space_size),
            keras.layers.LeakyReLU()
        ])

        if self.summary:
            print("###########")
            print("# ENCODER #")
            print("###########")
            print(self.encoder.summary())

    def build_decoder(self):
        self.decoder = keras.Sequential([
            keras.layers.Dense(64, input_shape = (self.latent_space_size,)),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(100),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(200),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(400),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(784),
            keras.layers.Activation("sigmoid"),
            keras.layers.Reshape(self.shape)
        ])

        if self.summary:
            print("###########")
            print("# DECODER #")
            print("###########")
            print(self.decoder.summary())

    def __init__(self, shape, latent_space_size, summary=False):
        self.shape = shape
        self.latent_space_size = latent_space_size
        self.summary = summary

        self.build_encoder()
        self.build_decoder()

        img = keras.Input(shape = self.shape)
        latent_vector = self.encoder(img)
        output = self.decoder(latent_vector)

        self.autoencoder = keras.Model(inputs = img, outputs = output)

        if self.summary:
            print("###############")
            print("# AUTOENCODER #")
            print("###############")
            print(self.autoencoder.summary())

        self.autoencoder.compile("nadam", loss="binary_crossentropy")

    def import_models(self, path, force=False, compile=True):
        root = Path(path)

        if not root.exists():
            if force:
                raise ValueError(f"directory {path} does not exist")
            else:
                return

        enc = Path(path+ENC_FILENAME)
        dec = Path(path+DEC_FILENAME)
        aut = Path(path+AUT_FILENAME)

        if not enc.exists():
            if force:
                raise ValueError(f"file {enc} does not exist")
            else:
                return
        if not dec.exists():
            if force:
                raise ValueError(f"file {dec} does not exist")
            else:
                return
        if not aut.exists():
            if force:
                raise ValueError(f"file {aut} does not exist")
            else:
                return

        self.encoder = keras.models.load_model(str(enc), compile=compile)
        self.decoder = keras.models.load_model(str(dec), compile=compile)
        self.autoencoder = keras.models.load_model(str(aut), compile=compile)

    def export(self, path):
        root = Path(path)

        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)

        self.encoder.save(os.path.normpath(path + ENC_FILENAME))
        self.decoder.save(os.path.normpath(path + DEC_FILENAME))
        self.autoencoder.save(os.path.normpath(path + AUT_FILENAME))

    def train(self, training_data, testing_data, epochs=10, batch_size=256):
        self.autoencoder.fit(
            training_data, training_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(testing_data, testing_data),
        )


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

    n = 5  # how many digits we will display

    plt.figure(figsize=(10,10)) # specifying the overall grid size

    for i in range(n*n):
        plt.subplot(n,n,i+1)    # the number of images in the grid is 5*5 (25)
        plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.axis('off')
        plt.gray()

    plt.show()


def train(vae):
    (x_train, x_test) = load_prepared_data()
    vae.train(x_train, x_test, 128, 256)
    vae.export("gan/degan/vae")


def test():
    (x_train, _) = load_prepared_data()

    show_sample(x_train[0].reshape(28, 28))


if __name__ == "__main__":
    # map the MNIST images (28x28 with only 1 channel) to a space of 32 floats
    vae = VAE((28, 28, 1), 128, summary=True)
    vae.import_models("gan/models/mnist/degan/ae2")

    # train(vae)
    visualize(vae)

    vae.export("gan/models/mnist/degan/ae2")