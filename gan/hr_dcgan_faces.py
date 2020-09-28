from gan import HR_DCGAN_MNIST

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

class HR_DCGAN_Faces(HR_DCGAN_MNIST):
    '''
    This class represents the network architecture for Face dataset from the paper:
    High-Resolution Deep Convolutional Generative Adversarial Networks
    J. D. Curtó, I. C. Zarza, Fernando de la Torre, Irwin King, Michael R. Lyu
    2020
    https://arxiv.org/abs/1711.06491
    '''

    def __init__(self, path='gan/models/faces/hr_dcgan/', show_training_results=True):
        super().__init__(path=path, show_training_results=show_training_results)

    def build_generator(self):
        noise_shape = (self.get_noise_dim(),)

        model = Sequential([

            layers.Dense(8*8*512, use_bias=False, input_shape=noise_shape),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Reshape((8,8,512)),
            # shape (8, 8, 512)

            layers.Conv2DTranspose(256, (4,4), strides=(1,1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (8, 8, 256)

            # stride 2 -> larger image
            # thiccness 64 -> channels
            layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (16, 16, 128)

            layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (32, 32, 64)

            layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (64, 64, 32)

            layers.Conv2DTranspose(16, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (128, 128, 16)

            layers.Conv2DTranspose(8, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (256, 256, 8)

            layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (512, 512, 3)

        ])

        return model

    def build_discriminator(self):
        img_shape = (512, 512, 3)
        
        model = Sequential([

            layers.Conv2D(16, (4,4), strides=(2,2), padding='same', input_shape=img_shape),
            layers.ELU(),
            # shape (256, 256, 16)

            layers.Conv2D(32, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (128, 128, 32)

            layers.Conv2D(64, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (64, 64, 64)

            layers.Conv2D(128, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (32, 32, 128)
            
            layers.Conv2D(256, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (16, 16, 256)

            layers.Conv2D(512, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (8, 8, 256)

            layers.Conv2D(1024, (4,4), strides=(1,1), padding='same'),
            # shape (8, 8, 1024)

            layers.Flatten(),
            layers.Dense(1)
        ])
                
        return model
