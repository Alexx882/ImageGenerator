import tensorflow as tf
import numpy as np
import os
import random

DISC_FILENAME = '/discriminator.h5'
GEN_FILENAME = '/generator.h5'

class GAN:

    def __init__(self, shape, batch_size=128, summary=False, f_save=None, train_combined=True):
        '''
        creates a GAN instance that can be trained to generate images in the specified size
        @params:
            shape                    - Required  : shape of the input/output images in the format channels_last
            batch_size               - Optional  : number of times, a training run in an epoch is executed. total number of individual training rounds is epochs*iterations_X
            summary                  - Optional  : should the summary of the models be printed to console
            f_save                   - Optional  : function that gets executed once per epoch. it is in the form (GAN, int) -> void
            train_combined           - Optional  : Implicit training with a combined network if true
        '''
        # apply const arguments
        self.width = shape[0]
        self.height = shape[1]
        self.channels = shape[2]
        self.batch_size = batch_size
        self.shape = (self.width, self.height, self.channels)
        self.summary = summary
        self.f_save = f_save
        
        self.initialized_discriminator = False
        self.initialized_generator = False
        self.initialized_combined_model = False
        self.has_training_data = False

        self.set_architectures(compile=train_combined)
    
    def set_trainable(self, model: tf.keras.Model, value: bool):
        '''Updates the trainable property for all layers in model to value.'''
        for layer in model.layers: 
            layer.trainable = value

    def set_architectures(self, compile=True):
        '''Applies the architectures defined in build_*() methods.'''

        self.optimizer = tf.keras.optimizers.Adam(1e-5) # tf.keras.optimizers.Adam(0.0001, 0.9)

        self.set_discriminator(self.build_discriminator())
        if compile:
            self.discriminator.compile(
                loss='binary_crossentropy',
                optimizer=self.optimizer,
                metrics=['accuracy']
            )

        self.set_generator(self.build_generator())
        if compile:
            self.generator.compile(
                loss='binary_crossentropy', 
                optimizer=self.optimizer,
                metrics=['accuracy']
            )

        if compile:
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

        z = tf.keras.Input(shape=(100,))
        img = self.generator(z)

        valid = self.discriminator(img)

        self.combined = tf.keras.Model(z, valid)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.combined.compile(loss=loss, optimizer=self.optimizer)
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

    def build_discriminator(self):
        '''
        classifies images and tries to detect "fake" ones from the generator

        builds a sample discriminator to be used in the GAN. The last layer has only 1 node and indicates if it is a fake 
        image (0) or a real image (1)
        '''
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=self.shape),
                tf.keras.layers.Dense(512),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(256),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ]
        )

        img = tf.keras.Input(shape=self.shape)
        validity = model(img)

        if self.summary:
            print("#################")
            print("# DISCRIMINATOR #")
            print("#################")
            print("INPUT SHAPE: "+str(self.shape))

            print(model.summary())

        return tf.keras.Model(img, validity)

    def build_generator(self):
        '''
        transforms random noise into an image
        
        builds a sample discriminator to be used in the GAN. The last layer has the shape of the image set in the constructor. 
        it contains the generated image.
        '''
        noise_shape = (100,)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, input_shape=noise_shape),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.BatchNormalization(momentum=0.8),
                tf.keras.layers.Dense(512),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.BatchNormalization(momentum=0.8),
                tf.keras.layers.Dense(1024),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.BatchNormalization(momentum=0.8),
                tf.keras.layers.Dense(np.prod(self.shape), activation='tanh'),
                tf.keras.layers.Reshape(self.shape),
            ]
        )

        if self.summary:
            print("#############")
            print("# GENERATOR #")
            print("#############")
            print("NOISE SHAPE: "+str(noise_shape))
            print(model.summary())

        noise = tf.keras.Input(shape=noise_shape)
        img = model(noise)

        return tf.keras.Model(noise, img)

#region Training

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

    def get_real_data_sample(self):
        '''Returns a sample of the real training data with size [self.batch_size / 2].'''
        np.random.shuffle(self.training_data)

        half_batch_size = int(self.batch_size / 2)
        batch_real = self.training_data[:half_batch_size, :, :, :]
        return batch_real

#region Explicit Training

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @staticmethod
    def discriminator_loss(real_output, generated_output):
        '''The discriminator loss function, where real output should be classified as 1 and generated as 0.'''
        return GAN.bce(tf.ones_like(real_output), real_output) \
            + GAN.bce(tf.zeros_like(generated_output), generated_output)

    @staticmethod
    def generator_loss(generated_output):
        '''The generator loss function, where generated output should be classified as 0.'''
        return GAN.bce(tf.ones_like(generated_output), generated_output)

    # @tf.function
    def train_step(self):
        # prepare real data and noise input
        real_data = self.get_real_data_sample()
        noise_vector = tf.random.normal(mean=0, stddev=1, shape=(real_data.shape[0], 100))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Predict images with G
            gen_data = self.generator(noise_vector, training=True)

            # Predict classes with D
            d_fake_data = self.discriminator(gen_data, training=True)
            d_real_data = self.discriminator(real_data, training=True)

            # Compute losses
            d_loss_value = GAN.discriminator_loss(real_output=d_real_data, generated_output=d_fake_data)
            g_loss_value = GAN.generator_loss(generated_output=d_fake_data)

        # Now that we have computed the losses, we can compute the gradients 
        # (using the tape) and optimize the networks
        gradients_of_discriminator = disc_tape.gradient(d_loss_value, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss_value, self.generator.trainable_variables)

        # Apply gradients to variables
        self.optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return d_loss_value, g_loss_value

    def train_explicit(self, epochs):
        for i in range(epochs):
            d_loss, g_loss = self.train_step()
            print(f"Iteration {i}: Discriminator loss: {d_loss}, Generator loss: {g_loss}")

#endregion Explicit Training

#region Combined Training

    def _train_discriminator(self, epoch, iterations):
        '''
        select a random half of the training data (real images) and train the discriminator on them (output 1)
        select an equally sized array of generated images (fake images) and train the discriminator on them (output 0)
        '''

        for i in range(iterations):
            batch1_real = self.get_real_data_sample()
            half_batch_size = batch1_real.shape[0]

            # creates a [half_batch_size, 100] array of noise
            noise = np.random.normal(0, 1, (half_batch_size, 100))
            batch_fake = self.generator.predict(noise)

            data = list(zip(batch1_real, np.ones(half_batch_size)))
            data.extend(list(zip(batch_fake, np.zeros(half_batch_size))))
            random.shuffle(data)

            features, labels = zip(*data)
            features = np.asarray(features)
            labels = np.asarray(labels)

            # unfreeze discriminator for training
            self.set_trainable(self.generator, False)
            self.set_trainable(self.discriminator, True)
            d_loss = self.discriminator.train_on_batch(features, labels)
            self.set_trainable(self.discriminator, False)

            print(f"EPOCH {epoch+1}.{i+1} [D] loss: {d_loss[0]} ; acc: {100*d_loss[1]}")

    def _train_generator(self, epoch, iterations):
        '''
        create a vector of noise and feed it into the combined model with the aim 
        to get a classification of 1 (="real")
        '''

        for i in range(iterations):
            # shuffled for discriminator
            batch_size = self.batch_size
            noise = np.random.normal(0, 1, (batch_size, 100))

            target_y = np.array([.8] * batch_size) # should be detected as 1 for generator training

            # freeze discriminator while generator is trained
            self.set_trainable(self.discriminator, False)
            self.set_trainable(self.generator, True)
            g_loss = self.combined.train_on_batch(noise, target_y)
            self.set_trainable(self.generator, False)

            # print(f"EPOCH {epoch+1}.{i+1} [G] loss: {g_loss[0]} ; acc: {100*g_loss[1]}")
            print(f"EPOCH {epoch+1}.{i+1} [G] loss: {g_loss}")

    def train(self, epochs=5, iterations_generator=20, iterations_discriminator=20):
        '''
        train discriminator/generator for the number of epochs. during each epoch, both are 
        trained iterations_X times as specified in the constructor. if a save function is specified, 
        it will be executed once per epoch

        :param epochs: number of epochs to train
        :param iterations_generator: iterations per epoch for gen
        :param iterations_discriminator: iterations per epoch for disc
        '''
        
        if self.f_save != None:
            self.f_save(self, 0)

        for epoch in range(epochs):
            self._train_discriminator(epoch, iterations_discriminator)
            self._train_generator(epoch, iterations_generator)

            if self.f_save != None:
                self.f_save(self, epoch+1)

#endregion Combined Training

#endregion Training

    def generate(self):
        '''
        samples a noise-vector and returns the outputs of the last layer of the generator
        '''
        noise = np.random.normal(0, 1, (1, 100))
        return self.generator.predict(noise)

#region Import / Export
    def export(self, path):
        '''
        exports the discriminator/generator to the specified location
        '''
        self.discriminator.trainable = True
        self.generator.trainable = True
        self.discriminator.save(os.path.normpath(path + DISC_FILENAME))
        self.generator.save(os.path.normpath(path + GEN_FILENAME))

    def import_(self, path, silent=False):
        ''' 
        Imports the disc/gen from the specified location. 
        
        :param path: 
        '''

        disc_path = os.path.normpath(path + DISC_FILENAME)
        gen_path = os.path.normpath(path + GEN_FILENAME)

        if not os.path.exists(disc_path) or not os.path.exists(gen_path):
            err = f"Did not find models at path: {path}"
            if silent:
                print(err)
                return
            else:
                raise ValueError(err)

        print("Loading models from file")
        self.set_discriminator(tf.keras.models.load_model(disc_path))
        self.set_generator(tf.keras.models.load_model(gen_path))
        self.bake_combined()
#endregion Import / Export
