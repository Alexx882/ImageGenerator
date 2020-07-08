import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path

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
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64),
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
            keras.layers.Dense(128),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512),
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

    def import_models(self, path, force=False):
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

        self.encoder = keras.models.load_model(str(enc))
        self.decoder = keras.models.load_model(str(dec))
        self.autoencoder = keras.models.load_model(str(aut))

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