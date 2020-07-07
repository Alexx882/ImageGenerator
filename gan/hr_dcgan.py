# from gan import GAN
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib as plt

# class HR_DCGAN(GAN):
#     '''
#     This class represents the network architecture from the paper:

#     High-Resolution Deep Convolutional Generative Adversarial Networks
#     <br />    
#     J. D. Curtó, I. C. Zarza, Fernando de la Torre, Irwin King, Michael R. Lyu
#     '''

#     def __init__(self):
#         pass

def build_generator():
    noise_shape = (100,)

    model = Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=noise_shape),
        layers.BatchNormalization(),
        layers.ELU(),
        layers.Reshape((7,7,256)),

        # input (7,7,256)
        layers.Conv2D(128, 4, strides=(1,1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ELU(),

        # stride 2 -> larger image
        # thiccness 128 -> channels
        layers.Conv2D(64, 4, strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ELU(),

        layers.Conv2D(1, 4, strides=(2,2), padding='same', use_bias=False, activation='tanh'),
        layers.BatchNormalization(),
        layers.ELU()

    ])

    return model

gen = build_generator()
noise = tf.random.normal([1,100])
img = gen(noise, training=false)
plt.imshow(generated_image[0, :,:,0], cmap='gray')


def build_discriminator(self):
    img_shape = (28, 28, 1)
    
    model = Sequential([

        layers.Conv2D(64, 4, strides=(2,2), padding='same', input_shape=img_shape),
        layers.ELU(),

        layers.Conv2D(128, 4, strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.ELU(),

        layers.Conv2D(256, 4, strides=(1,1), padding='same', activation='sigmoid'),

        layers.Flatten(),
        layers.Dense(1)

    ])

    return model