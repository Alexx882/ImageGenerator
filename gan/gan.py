import tensorflow as tf
from tensorflow import keras
import numpy as np


class GAN:
    def __init__(self, shape, epochs=5, batch_size=128, iterations_generator=20, iterations_discriminator=20, summary=False, f_save=None):
        self.width = shape[0]
        self.height = shape[1]
        self.channels = shape[2]
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations_generator = iterations_generator
        self.iterations_discriminator = iterations_discriminator
        self.shape = (self.width, self.height, self.channels)
        self.summary = summary
        self.f_save = f_save

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = keras.Input(shape=(100,))
        img = self.generator(z)

        self.discriminator.trainable = False
        valid = self.discriminator(img)

        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    #
    # classifies images and tries to detect "fake" ones from the generator
    #
    def build_discriminator(self):
        model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=self.shape),
                keras.layers.Dense(512),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Dense(256),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Dense(1, activation='sigmoid'),
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

    #
    # transforms random noise into an image
    #
    def build_generator(self):
        noise_shape = (100,)

        model = keras.Sequential(
            [
                keras.layers.Dense(256, input_shape=noise_shape),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.BatchNormalization(momentum=0.8),
                keras.layers.Dense(512),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.BatchNormalization(momentum=0.8),
                keras.layers.Dense(1024),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.BatchNormalization(momentum=0.8),
                keras.layers.Dense(np.prod(self.shape), activation='tanh'),
                keras.layers.Reshape(self.shape),
            ]
        )

        if self.summary:
            print("#############")
            print("# GENERATOR #")
            print("#############")
            print(model.summary())

        noise = keras.Input(shape=noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

    def set_training_data(self, data, labels):
        self.training_data = data
        self.training_labels = labels

    def train_discriminator(self, epoch):
        '''
        select a random half of the training data (real images) and train the discriminator on them (output 1)
        select an equally sized array of generated images (fake images) and train the discriminator on them (output 0)
        '''

        for i in range(self.iterations_discriminator):
            np.random.shuffle(self.training_data)

            half_batch_size = int(self.batch_size / 2)
            batch1_real = self.training_data[:half_batch_size, :, :, :]
            # batch2_real = self.training_data[:half_batch_size, :, :, :] # other half of the samples

            # creates a half_batch_size|100 array of noise
            noise = np.random.normal(0, 1, (half_batch_size, 100))

            batch_fake = self.generator.predict(noise)

            # unfreeze discriminator for training
            self.discriminator.trainable = True

            d_loss_real = self.discriminator.train_on_batch(
                batch1_real, np.ones(half_batch_size))
            d_loss_fake = self.discriminator.train_on_batch(
                batch_fake, np.zeros(half_batch_size))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            self.discriminator.trainable = False

            print("EPOCH %d.%d [D] loss: %f, acc.: %.2f%%]" %
                (epoch+1, i+1, d_loss[0], 100*d_loss[1]))

    def train_generator(self, epoch):
        '''
        create a vector of noise and feed it into the combined model with the aim 
        to get a classification of 1 (="real")
        '''

        for i in range(self.iterations_generator):
            batch_size = self.training_data.shape[0]
            noise = np.random.normal(0, 1, (batch_size, 100))

            valid_y = np.array([1] * batch_size)
            # freeze discriminator while generator is trained
            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch(noise, valid_y)

            print("EPOCH %d.%d [G] loss: %f]" % (epoch+1, i+1, g_loss))


    def train(self):
        '''
        train discriminator/generator for the number of epochs
        '''

        if self.f_save != None:
                self.f_save(self, 0)

        for epoch in range(self.epochs):
            self.train_discriminator(epoch)
            self.train_generator(epoch)

            if self.f_save != None:
                self.f_save(self, epoch+1)
            
    def generate(self):
        noise = np.random.normal(0, 1, (1, 100))
        return self.generator.predict(noise)

    def predict(self):
        # batchnr | image width | image height | channel
        data = np.arange(28*28).reshape(1, 28, 28, 1)

        return self.discriminator.predict(data)


if __name__ == "__main__":
    gan = GAN((28, 28, 1))
    print(gan.predict())
