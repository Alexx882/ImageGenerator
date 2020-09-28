from gan.gan import GAN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

from gan.degan_ae import AE_MNIST

class DEGAN_MNIST(GAN):
    '''
    This class represents the network architecture for MNIST dataset from the paper: 
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    Alec Radford, Luke Metz, Soumith Chintala
    2016
    https://arxiv.org/abs/1511.06434
    '''

    def __init__(self, path='gan/models/mnist/degan/', show_training_results=True):
        super().__init__(path=path, show_training_results=show_training_results)
        self.ae = AE_MNIST(128, path="ae_degan_mnist")

    def build_generator(self):
        n_nodes = 128 * 7 * 7

        model = Sequential([
            layers.Dense(n_nodes, input_dim=128),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, 128)),
            # upsample to 14x14
            layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            # upsample to 28x28
            layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7,7), activation='sigmoid', padding='same')
        ], name="generator")

        return model

    def sample_noise(self, batch_size, noise_shape):
        noise = super().sample_noise(batch_size, noise_shape)

        decoded = self.ae.decoder(noise)
        encoded = self.ae.encoder(decoded)

        return encoded

    def build_discriminator(self):
        img_shape = (28, 28, 1)
        
        model = Sequential([
            layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=img_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),
            layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def get_noise_dim(self):
        return 128

    def get_optimizers(self):
        d = tf.keras.optimizers.Adam(.0002, .5) 
        g = tf.keras.optimizers.Adam(.0002, .5) 
        return (d, g)