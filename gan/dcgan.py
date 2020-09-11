from gan.gan import GAN

import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class DCGAN(GAN):
    '''
    This class represents the network architecture from the paper:

    High-Resolution Deep Convolutional Generative Adversarial Networks
    <br />    
    J. D. Curtó, I. C. Zarza, Fernando de la Torre, Irwin King, Michael R. Lyu
    '''

    def __init__(self, shape, f_save=None, train_combined=False):
        super().__init__(shape, f_save=f_save, train_combined=train_combined)

    def build_generator(self):
        noise_shape = (100,)

        model = Sequential([
            layers.Dense(7*7*256, use_bias=False, input_shape=noise_shape),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Reshape((7,7,256)),
            # shape (7,7,256)

            layers.Conv2DTranspose(128, (4,4), strides=(1,1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # shape (7,7,128)

            # stride 2 -> larger image
            # thiccness 128 -> channels
            layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # shape (14,14,64)

            layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            # shape (28,28,1)

        ])

        return model
        # noise = keras.Input(shape=noise_shape)
        # img = model(noise)

        # return keras.Model(noise, img)

# gen = build_generator()
# noise = tf.random.normal([1,100])
# img = gen(noise, training=False)
# print(img.shape)
# plt.imshow(img[0, :,:,0], cmap='gray')
# plt.show()


    def build_discriminator(self):
        img_shape = (28, 28, 1)
        
        model = Sequential([

            layers.Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=img_shape),
            layers.LeakyReLU(),

            layers.Conv2D(128, (4,4), strides=(2,2), padding='same'),
            layers.LeakyReLU(),

            # layers.Conv2D(256, 4, strides=(1,1), padding='same', activation='sigmoid'),

            layers.Flatten(),
            layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        
        return model
        # img = keras.Input(shape=img_shape)
        # validity = model(img)
        
        # return keras.Model(img, validity)

# disc = build_discriminator()
# res = disc.predict(img)
# print(res)

