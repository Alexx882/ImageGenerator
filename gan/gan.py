import tensorflow as tf
import numpy as np
import os
import random
from pathlib import Path
from collections import deque
import time
from IPython import display
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import imageio
import glob 
from typing import Tuple, Iterable, Callable

DISC_FILENAME = '/discriminator.h5'
GEN_FILENAME = '/generator.h5'

class GAN(ABC):

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, path, show_training_results=True):
        '''
        creates a GAN instance that can be trained to generate images in the specified size
        @params:
            path                     - Required  : location on disc to store progress images and model on export 
            show_training_results    - Optional  : shows intermediate results with plt.show() if true
        '''
        self._set_path(path)
        
        self._init_training_visualization(show_training_results)
        
        self.has_training_data = False
        self._set_architecture()

    def _set_path(self, path):
        self.path = path
        Path(os.path.normpath(path + '/images/')).mkdir(parents=True, exist_ok=True)

    def _init_training_visualization(self, show_training_results):
        self.show_training_results = show_training_results
        self.num_examples_to_generate = 16
        self.seed = tf.random.normal([self.num_examples_to_generate, self.get_noise_dim()])

    def _set_architecture(self):
        '''Applies the architecture from the implementing strategy.'''
        d_o, g_o = self.get_optimizers()
        self.d_optimizer = d_o
        self.g_optimizer = g_o

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

#region Neural Network Details

    @abstractmethod
    def build_discriminator(self) -> tf.keras.Model:
        '''This method must return the model for the Discriminator.'''
        pass

    @abstractmethod
    def build_generator(self) -> tf.keras.Model:
        '''This method must return the model for the Generator.'''
        pass

    @abstractmethod
    def get_noise_dim(self) -> int:
        '''This method must return the 1d size of the noise, eg. 100.'''
        pass

    @abstractmethod
    def get_optimizers(self) -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.Optimizer]:
        '''This class must return two optimizers for Discriminator and Generator in this order.'''
        pass
    
#endregion Neural Network Details

#region Training

    def set_training_data(self, data_generator_func: Callable[[], Iterable[np.ndarray]]):
        '''
        sets the training data that should be used.
        @params:
            data_generator_func. a generator function which yields real training images in the desired batch size on every call.
                                 Here we use a generator function, as a normal generator would stop yielding results after loading all values once.
                                 With the function we can execute the function everytime to get the same values again in the next epoch. 
                                 Random order is preferred.
        '''
        # training data. the shape should have one dimension more (in the beginning) than the image shape 
        # of the network. This dimension indicates the individual images. f.e. if images are 28x28 pixel 
        # with only one channel, the shape of the training data should be (number_of_rows, 28, 28, 1)

        self.training_data_generator_func = data_generator_func
        self.has_training_data = True

    @staticmethod
    def discriminator_loss(real_output, generated_output):
        '''The discriminator loss function, where real output should be classified as 1 and generated as 0.'''
        return GAN.bce(tf.ones_like(real_output), real_output) + GAN.bce(tf.zeros_like(generated_output), generated_output)

    @staticmethod
    def generator_loss(generated_output):
        '''The generator loss function, where generated output should be classified as 0.'''
        return GAN.bce(tf.ones_like(generated_output), generated_output)

    def sample_noise(self, batch_size, noise_shape):
        noise = tf.random.normal([batch_size, self.get_noise_dim()])
        return noise

    @tf.function
    def train_step(self, real_data_batch: np.ndarray) -> '(disc_loss, gen_loss)':
        # prepare real data and noise input
        batch_size = real_data_batch.shape[0]
        noise = self.sample_noise(batch_size, self.get_noise_dim)

        print(noise.shape)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Predict images with G
            gen_data = self.generator(noise, training=True)

            # Predict classes with D
            d_real_predicted_labels = self.discriminator(real_data_batch, training=True)
            d_fake_predicted_labels = self.discriminator(gen_data, training=True)

            # Compute losses
            d_loss_value = GAN.discriminator_loss(real_output=d_real_predicted_labels, generated_output=d_fake_predicted_labels)
            g_loss_value = GAN.generator_loss(generated_output=d_fake_predicted_labels)

        # Now that we have computed the losses, we can compute the gradients (using the tapes)
        gradients_of_discriminator = disc_tape.gradient(d_loss_value, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss_value, self.generator.trainable_variables)

        # Apply gradients to variables
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return d_loss_value, g_loss_value

    def train(self, epochs):
        if not self.has_training_data:
            raise RuntimeError("Training Data is not set.")

        for epoch in range(epochs):
            start = time.time()

            for image_batch in self.training_data_generator_func():
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self._generate_and_save_images(self.generator, epoch + 1, self.seed)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self._generate_and_save_images(self.generator, epochs, self.seed)

    def _generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(os.path.normpath(self.path + '/images/image_at_epoch_{:04d}.png'.format(epoch)))
        if self.show_training_results:
            plt.show()

    def generate_gif(self, extend_last_frame=True):
        import tensorflow_docs.vis.embed as embed # pip install git+https://github.com/tensorflow/docs

        anim_file = os.path.normpath(self.path + '/progress.gif')

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(os.path.normpath(self.path + '/images/image*.png'))
            filenames = sorted(filenames)

            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
            for i in range(24 if extend_last_frame else 1):
                image = imageio.imread(filename)
                writer.append_data(image)

        embed.embed_file(anim_file)

    def train_explicit(self, epochs):
        # TODO check if needed and fix
        window_size = 5+1
        train_disc = train_gen = True
        iterations = 20
        # calculate a floating window to help decide number of iterations
        disc_losses_window = deque(maxlen=window_size)
        gen_losses_window = deque(maxlen=window_size)

        cur_epoch = 0
        while cur_epoch < epochs:

            for _ in range(iterations):
                current_disc_loss, current_gen_loss = self.train_step(train_disc, train_gen)

            print(f"[E {cur_epoch}] Trained {iterations} iterations: Final Discriminator loss: {current_disc_loss}, Final Generator loss: {current_gen_loss}")
            disc_losses_window.append(current_disc_loss)
            gen_losses_window.append(current_gen_loss)

            # decide for the next training:
            # if both got worse or got better -> (continue to) train both
            # if only one part got worse -> just train that part
            disc_got_worse: bool = current_disc_loss >= max(disc_losses_window)
            gen_got_worse: bool = current_gen_loss >= max(gen_losses_window)

            train_disc = train_gen = True
            iterations = 20

            if not (disc_got_worse ^ gen_got_worse):
                cur_epoch += 1

            if disc_got_worse and not gen_got_worse:
                train_disc = True
                train_gen = False
                iterations = 5
                print(f"Discriminator got worse, explicit training with {iterations} iterations")
            if gen_got_worse and not disc_got_worse:
                train_disc = False
                train_gen = True
                iterations = 5
                print(f"Generator got worse, explicit training with {iterations} iterations")

    def generate(self):
        '''
        samples a noise-vector and returns the outputs of the last layer of the generator
        '''
        noise = np.random.normal(0, 1, (1, 100))
        return self.generator.predict(noise)

#endregion Training

#region Import / Export

    def export(self):
        '''
        exports the discriminator/generator to the specified location
        '''
        self.discriminator.trainable = True
        self.generator.trainable = True
        self.discriminator.save(os.path.normpath(self.path + DISC_FILENAME))
        self.generator.save(os.path.normpath(self.path + GEN_FILENAME))

    def import_(self, path=None, silent=False):
        ''' 
        Imports the disc/gen from the specified location. 
        
        :param path: 
        '''
        if path is None:
            path = self.path

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
        self.discriminator = (tf.keras.models.load_model(disc_path))
        self.generator = (tf.keras.models.load_model(gen_path))

    @staticmethod
    def import_gan(path, silent=False):
        gan = _GAN()
        gan.import_(path, silent)
        return gan

#endregion Import / Export


class _GAN(GAN):
    '''
    This class is used to import unknown trained models from file.
    It must only be used for GAN.import_gan(path, silent) and provides almost no functionality.
    '''

    def __init__(self):
        super().__init__(path='', show_training_results=False)
        self.error = NotImplementedError("This class is only used for generic imported models")

    def _set_path(self, path):
        pass

    def _init_training_visualization(self, show_training_results):
        pass

    def _set_architecture(self):
        pass

    def train(self, epochs):
        raise self.error

    def build_generator(self):
        raise self.error

    def build_discriminator(self):
        raise self.error

    def get_noise_dim(self):
        print(self.generator.input_shape)
        return self.generator.input_shape

    def get_optimizers(self):
        raise self.error

        