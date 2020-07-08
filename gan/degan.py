import gan
import tensorflow as tf
from tensorflow import keras
import numpy as np

class DEGAN(gan.GAN):
    def __init__(self, shape, f_save=None, summary=False, noise_size=100):
        super().__init__(shape, f_save=f_save, summary=summary, noise_size=noise_size)
    
    def bake_combined(self):
        if not self.initialized_discriminator or not self.initialized_generator:
            raise ValueError("Generator/Discriminator not initialized yet!")

        z = keras.Input(shape=(self.noise_size,))
        img = self.generator(z)

        valid = self.discriminator(img)

        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.initialized_combined_model = True

    def build_discriminator(self):
        model = keras.Sequential(
            [
                keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=self.shape),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Dropout(0.4),
                keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=self.shape),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Dropout(0.4),
                keras.layers.Flatten(),
                keras.layers.Dense(1, activation='sigmoid')
            ]
        )

        img = keras.Input(shape=self.shape)
        validity = model(img)

        if self.summary:
            print("#################")
            print("# DISCRIMINATOR #")
            print("#################")
            print("INPUT SHAPE: "+str(self.shape))

            print(model.summary())

        return keras.Model(img, validity)

    def build_generator(self):
        noise_shape = (128,)

        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7

        model = keras.Sequential(
            [        
                keras.layers.Dense(n_nodes, input_dim=self.noise_size),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Reshape((7, 7, 128)),
                # upsample to 14x14
                keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.2),
                # upsample to 28x28
                keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(1, (7,7), activation='sigmoid', padding='same'),
            ]
        )

        noise = keras.Input(shape=noise_shape)
        img = model(noise)

        if self.summary:
            print("#############")
            print("# GENERATOR #")
            print("#############")
            print("NOISE SHAPE: "+str(noise_shape))
            print(model.summary())

        return keras.Model(noise, img)

if __name__ == "__main__":
    # instantiate on images from MNIST
    degan = DEGAN((28,28,1), summary=True, noise_size=128)

    # labels are not needed as the GAN only distinguishes between "real" and "fake"
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # bring x_train into 4d shape (id, row, col, channel) and normalize values to [0;1]
    x_train = np.array([np.array(sample).reshape((28,28,1)) for sample in x_train]) / 255.
    
    degan.set_training_data(x_train)
    degan.doctor()
    degan.train()