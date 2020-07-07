import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist


class VAE:
    def __init__(self, shape, latent_space_size, summary = False):
        self.shape = shape
        self.latent_space_size = latent_space_size

        self.summary = summary

        # self.build_encoder()
        # self.build_decoder()
        
        image_input = keras.Input(shape=self.shape, name="input_autoencoder")
        l1 = keras.layers.Flatten(input_shape=self.shape, name="flat_source")(image_input)
        latent_layer = keras.layers.Dense(self.latent_space_size, activation="relu", name="latent_layer")(l1)
        l2 = keras.layers.Dense(784, activation="relu")(latent_layer)
        image_output = keras.layers.Reshape(self.shape)(l2)

        self.autoencoder = keras.Model(image_input, image_output)
        if self.summary:
            print("###############")
            print("# AUTOENCODER #")
            print("###############")
            print(self.autoencoder.summary())
        
        input_encoder = keras.Input(shape=self.shape, name="input_encoder")
        l5 = self.autoencoder.layers[1](input_encoder)
        l6 = self.autoencoder.layers[2](l5)

        self.encoder = keras.Model(input_encoder, l6)
        if self.summary:
            print("###########")
            print("# ENCODER #")
            print("###########")
            print(self.encoder.summary())

        latent_input = keras.Input(shape=(self.latent_space_size))
        l3 = self.autoencoder.layers[-2](latent_input)
        l4 = self.autoencoder.layers[-1](l3)

        self.decoder = keras.Model(latent_input, l4)
        if self.summary:
            print("###########")
            print("# DECODER #")
            print("###########")
            print(self.decoder.summary())
        
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

if __name__ == "__main__":
    # map the MNIST images (28x28 with only 1 channel) to a space of 32 floats
    vae = VAE((28, 28, 1), 32)

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    print(x_train.shape)

    vae.autoencoder.fit(
        x_train, x_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test,x_test),
    )
