import tensorflow as tf
from tensorflow import keras
import numpy as np


class GAN:
    def __init__(self, shape, epochs=5, batch_size=128, iterations_generator=20, iterations_discriminator=20, summary=False, f_save=None):
        '''
        creates a GAN instance that can be trained to generate images in the specified size
        @params:
            shape                    - Required  : shape of the input/output images in the format channels_last
            epochs                   - Optional  : number of times, the generator/discriminator are trained. 
            batch_size               - Optional  : number of times, a training run in an epoch is executed. total number of individual training rounds is epochs*iterations_X
            iterations_generator     - Optional  : how many iterations are executed for the generator
            iterations_discriminator - Optional  : how many iterations are executed for the discriminator
            summary                  - Optional  : should the summary of the models be printed to console
            f_save                   - Optional  : function that gets executed once per epoch. it is in the form (GAN, int) -> void
        '''
        # apply const arguments
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
        
        self.initialized_discriminator = False
        self.initialized_generator = False
        self.initialized_combined_model = False
        self.has_training_data = False

        self.set_architectures()
    
    def set_architectures(self):
        '''Applies the architectures defined in build_*() methods.'''

        self.optimizer = keras.optimizers.Adam(0.0002, 0.5)

        self.set_discriminator(self.build_discriminator())
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        # self.discriminator.trainable = False

        self.set_generator(self.build_generator())
        self.generator.compile(
            loss='binary_crossentropy', 
            optimizer=self.optimizer
        )
        # self.generator.trainable = False

        self.bake_combined()
    
    def set_discriminator(self, model):
        '''
        specifies the discriminator of the GAN network. sets initialized_discriminator to True which is necessary for baking the combined model
        @params:
            model - Required : network architecture for the discriminator
        '''
        self.discriminator = model
        self.initialized_discriminator = True

    def set_generator(self, model):
        '''
        specifies the generator of the GAN network. sets initialized_generator to True which is necessary for baking the combined model
        @params:
            model - Required : network architecture for the generator
        '''
        self.generator = model
        self.initialized_generator = True
    
    def bake_combined(self):
        '''
        specifies the generator of the GAN network. sets initialized_generator to True which is necessary for baking the combined model
        @params:
            model - Required : network architecture for the generator
        '''
        if not self.initialized_discriminator or not self.initialized_generator:
            raise ValueError("Generator/Discriminator not initialized yet!")

        z = keras.Input(shape=(100,))
        img = self.generator(z)

        valid = self.discriminator(img)

        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.initialized_combined_model = True

    def doctor(self):
        '''
        performs a self-check to debug if all necessary actions were done before starting to train/use the network
        '''

        print("##############")
        print("# GAN DOCTOR #")
        print("##############")

        s = "Initialized generator"

        if self.initialized_generator:
            print(s + " ... OK")
        else:
            print(s+" ... NOK")
        
        s = "Initialized discriminator"

        if self.initialized_discriminator:
            print(s + " ... OK")
        else:
            print(s+" ... NOK")
        
        s = "Baked combined model"
        if self.initialized_combined_model:
            print(s + " ... OK")
        else:
            print(s+" ... NOK")
        
        s = "Has training data"
        if self.has_training_data:
            print(s + " ... OK")
        else:
            print(s+" ... NOK")

    #
    # classifies images and tries to detect "fake" ones from the generator
    #
    def build_discriminator(self):
        '''
        builds a sample discriminator to be used in the GAN. The last layer has only 1 node and indicates if it is a fake 
        image (0) or a real image (1)
        '''
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
        '''
        builds a sample discriminator to be used in the GAN. The last layer has the shape of the image set in the constructor. 
        it contains the generated image.
        '''
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
            print("NOISE SHAPE: "+str(noise_shape))
            print(model.summary())

        noise = keras.Input(shape=noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

    def set_training_data(self, data):
        '''
        sets the training data that should be used.
        @params:
            data - Required : training data. the shape should have one dimension more (in the beginning) than the image shape 
                              of the network. This dimension indicates the individual images. f.e. if images are 28x28 pixel 
                              with only one channel, the shape of the training data should be (number_of_rows, 28, 28, 1)
        '''

        self.training_data = data
        self.has_training_data = True

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

            valid_y = np.array([0.9] * batch_size)
            # freeze discriminator while generator is trained
            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch(noise, valid_y)
            # self.generator.trainable = False

            print("EPOCH %d.%d [G] loss: %f]" % (epoch+1, i+1, g_loss))

    def train(self):
        '''
        train discriminator/generator for the number of epochs. during each epoch, both are 
        trained iterations_X times as specified in the constructor. if a save function is specified, 
        it will be executed once per epoch
        '''
        
        if self.f_save != None:
            self.f_save(self, 0)

        for epoch in range(self.epochs):
            self.train_discriminator(epoch)
            self.train_generator(epoch)

            if self.f_save != None:
                self.f_save(self, epoch+1)

    def generate(self):
        '''
        samples a noise-vector and returns the outputs of the last layer of the generator
        '''
        noise = np.random.normal(0, 1, (1, 100))
        return self.generator.predict(noise)

    def export(self, path):
        '''
        exports the discriminator/generator to the specified location
        '''
        self.discriminator.trainable = True
        self.generator.trainable = True
        self.discriminator.save(path+"/discriminator.h5")
        self.generator.save(path+"/generator.h5")
