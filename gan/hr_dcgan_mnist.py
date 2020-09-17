from gan.gan import GAN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

class HR_DCGAN_MNIST(GAN):
    '''
    This class represents the network architecture for MNIST dataset from the paper:
    High-Resolution Deep Convolutional Generative Adversarial Networks
    J. D. Curtó, I. C. Zarza, Fernando de la Torre, Irwin King, Michael R. Lyu
    2020
    https://arxiv.org/abs/1711.06491
    '''

    def __init__(self, path='gan/models/mnist/hr_dcgan/'):
        super().__init__(path=path)

    def build_generator(self):
        noise_shape = (self.get_noise_dim(),)

        model = Sequential([

            # project and reshape
            layers.Dense(7*7*256, use_bias=False, input_shape=noise_shape),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Reshape((7,7,256)),
            # shape (7, 7, 256)

            layers.Conv2DTranspose(128, (4,4), strides=(1,1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (7, 7, 128)

            # stride 2 -> larger image
            # thiccness 64 -> channels
            layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (14, 14, 64)

            layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (28, 28, 1)

        ])

        return model

    def build_discriminator(self):
        img_shape = (28, 28, 1)
        
        model = Sequential([

            layers.Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=img_shape),
            layers.ELU(),
            # shape (14, 14, 64)

            layers.Conv2D(128, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (7, 7, 128)

            layers.Conv2D(256, (4,4), strides=(1,1), padding='same'),
            # shape (7, 7, 256)

            layers.Flatten(),
            layers.Dense(1) #, activation='sigmoid')
            # FIXME see DCGAN why sigmoid is not used
        ])
                
        return model

    def get_noise_dim(self):
        return 100

    def get_optimizers(self):
        return (
            tf.keras.optimizers.Adam(.0002, .5), 
            tf.keras.optimizers.Adam(.0002, .5)
        )