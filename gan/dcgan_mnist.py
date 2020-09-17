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
    2016
    https://arxiv.org/abs/1511.06434
    '''

    def __init__(self):
        super().__init__(path='gan/models/mnist/dcgan/', batch_size=128)

    def build_generator(self):
        noise_shape = (self.get_noise_dim(),)

        model = Sequential([

            # project and reshape
            layers.Dense(7*7*128, use_bias=False, input_shape=noise_shape),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((7, 7, 128)),
            # shape (7, 7, 128)

            # stride 2 -> larger image
            # thiccness 64 -> channels
            layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # shape (14, 14, 64)

            layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (28, 28, 1)

        ])

        return model

    def build_discriminator(self):
        img_shape = (28, 28, 1)
        
        model = Sequential([

            layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=img_shape),
            layers.LeakyReLU(alpha=.2),
            # shape (14, 14, 64)

            layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=.2),
            # shape (7, 7, 128)

            layers.Flatten(),
            layers.Dense(1) #, activation='sigmoid')
            # FIXME when using sigmoid as proposed by the paper the classification does not work.
            # somehow the  two classes (real/fake) cannot be differentiated
            # I assume this has to do with sigmoid being independent between classes (and real/fake being not)
            # https://gombru.github.io/2018/05/23/cross_entropy_loss/
        ])
        
        return model

    def get_noise_dim(self):
        return 100

    def get_optimizers(self):
        d = tf.keras.optimizers.Adam(.0002, .5) 
        g = tf.keras.optimizers.Adam(.0002, .5) 
        return (d, g)