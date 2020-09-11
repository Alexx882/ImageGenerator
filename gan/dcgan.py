from gan.gan import GAN

import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

class DCGAN(GAN):
    '''
    This class represents the network architecture from the paper: 
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    Alec Radford, Luke Metz, Soumith Chintala
    https://arxiv.org/abs/1511.06434
    '''

    def __init__(self):
        # TODO batch_size = 128
        # TODO leaky relu = .2
        # TODO no obvious dropout
        super().__init__(path='gan/models/dcgan/')

    def build_generator(self):
        noise_shape = (100,)

        model = Sequential([
            layers.Dense(7*7*256, use_bias=False, input_shape=noise_shape),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Reshape((7,7,256)),
            # shape (7,7,256)

            layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # shape (7,7,128)

            # stride 2 -> larger image
            # thiccness 128 -> channels
            layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # shape (14,14,64)

            layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (28,28,1)

        ])

        return model

    def build_discriminator(self):
        img_shape = (28, 28, 1)
        
        model = Sequential([

            layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=img_shape),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            # layers.Conv2D(256, 4, strides=(1,1), padding='same', activation='sigmoid'),

            layers.Flatten(),
            layers.Dense(1) #, activation=tf.nn.sigmoid)
        ])
        
        return model

    def get_noise_dim(self):
        return 100

    def get_optimizers(self):
        d = tf.keras.optimizers.Adam(.0002, .5) 
        g = tf.keras.optimizers.Adam(.0002, .5) 
        return (d, g)