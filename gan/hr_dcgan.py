from gan.gan import GAN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class HR_DCGAN(GAN):
    '''
    This class represents the network architecture from the paper:

    High-Resolution Deep Convolutional Generative Adversarial Networks
    <br />    
    J. D. Curtó, I. C. Zarza, Fernando de la Torre, Irwin King, Michael R. Lyu
    '''

    def __init__(self, shape, batch_size=128, f_save=None):
        super().__init__(shape, batch_size=128, f_save=f_save)

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
            # TODO read about Conv2DTranspose
            layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # shape (14,14,64)

            layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (28,28,1)

        ])

        noise = keras.Input(shape=noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

# gen = build_generator()
# noise = tf.random.normal([1,100])
# img = gen(noise, training=False)
# print(img.shape)
# plt.imshow(img[0, :,:,0], cmap='gray')
# plt.show()


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
        
        img = keras.Input(shape=img_shape)
        validity = model(img)
        
        return keras.Model(img, validity)

# disc = build_discriminator()
# res = disc.predict(img)
# print(res)

