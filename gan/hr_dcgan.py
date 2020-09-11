from gan.gan import GAN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

class HR_DCGAN(GAN):
    '''
    This class represents the network architecture from the paper:
    High-Resolution Deep Convolutional Generative Adversarial Networks
    J. D. Curtó, I. C. Zarza, Fernando de la Torre, Irwin King, Michael R. Lyu
    https://arxiv.org/abs/1711.06491
    '''

    def __init__(self):
        # TODO remove initial generator dense layer part (start with conv)
        super().__init__(path='gan/models/hr_dcgan', batch_size=128)

    def build_generator(self):
        noise_shape = (100,)

        model = Sequential([
            layers.Dense(7*7*256, use_bias=False, input_shape=noise_shape),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Reshape((7,7,256)),
            # shape (7,7,256)

            layers.Conv2DTranspose(128, (4,4), strides=(1,1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (7,7,128)

            # stride 2 -> larger image
            # thiccness 128 -> channels
            layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (14,14,64)

            layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (28,28,1)

        ])

        return model

    def build_discriminator(self):
        img_shape = (28, 28, 1)
        
        model = Sequential([

            layers.Conv2D(64, 4, strides=(2,2), padding='same', input_shape=img_shape),
            layers.ELU(),

            layers.Conv2D(128, 4, strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ELU(),

            # layers.Conv2D(256, 4, strides=(1,1), padding='same', activation='sigmoid'),

            layers.Flatten(),
            layers.Dense(1)
        ])
                
        return model

    def get_noise_dim(self):
        return 100

    def get_optimizers(self):
        return (tf.keras.optimizers.Adam(.0002, .5), tf.keras.optimizers.Adam(.0002, .5))