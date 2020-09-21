import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from gan.degan_mnist import DEGAN_MNIST
from herold.mnist import MNIST_provider

def test_provider():
    provider = MNIST_provider(1000)

    i = 0

    for batch in provider.get_numbers():
        print(f"batch {i} ... {batch.shape}")
        i += 1

def mnist_training():
    epochs = 1
    provider = MNIST_provider(128)
    gan = DEGAN_MNIST()

    gan.set_training_data(provider.get_numbers)
    
    for epoch in range(epochs):
        i = 0

        # start training
        for image_batch in provider.get_numbers():
            print(f"batch {i}")
            gan.train_step(image_batch)
            i += 1
        gan._generate_and_save_images(gan.generator, epoch, gan.seed)
        # end training

    gan.export()
    # gan.generate_gif()
    

if __name__ == "__main__":
    mnist_training()
    
    

