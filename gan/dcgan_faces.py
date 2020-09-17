from gan import DCGAN_MNIST

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

class DCGAN_Faces(DCGAN_MNIST):
    '''
    This class represents the network architecture for MNIST dataset from the paper: 
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    Alec Radford, Luke Metz, Soumith Chintala
    2016
    https://arxiv.org/abs/1511.06434
    '''

    def __init__(self, path='gan/models/faces/dcgan/', show_training_results=True):
        super().__init__(path=path, show_training_results=show_training_results)

    def build_generator(self):
        noise_shape = (self.get_noise_dim(),)

        model = Sequential([

            # project and reshape
            layers.Dense(8*8*256, use_bias=False, input_shape=noise_shape),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((8, 8, 256)),
            # shape (8, 8, 256)

            # stride 2 -> larger image
            # thiccness 128 -> channels
            # layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
            # layers.BatchNormalization(),
            # layers.ReLU(),
            # shape (16, 16, 128)
            
            layers.Conv2DTranspose(64, (5,5), strides=(4,4), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # shape (32, 32, 64)
            
            # layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False),
            # layers.BatchNormalization(),
            # layers.ReLU(),
            # shape (64, 64, 32)

            layers.Conv2DTranspose(16, (5,5), strides=(4,4), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # shape (128, 128, 16)

            layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (256, 256, 1)

        ])

        return model

    def build_discriminator(self):
        img_shape = (256, 256, 1)
        
        model = Sequential([

            layers.Conv2D(16, (5,5), strides=(2,2), padding='same', input_shape=img_shape),
            layers.LeakyReLU(alpha=.2),
            # shape (128, 128, 16)

            # layers.Conv2D(32, (5,5), strides=(2,2), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=.2),
            # shape (64, 64, 32)

            layers.Conv2D(64, (5,5), strides=(4,4), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=.2),
            # shape (32, 32, 64)

            # layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=.2),
            # shape (16, 16, 128)

            layers.Conv2D(256, (5,5), strides=(4,4), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=.2),
            # shape (8, 8, 256)

            layers.Flatten(),
            layers.Dense(1)
        ])
        
        return model
