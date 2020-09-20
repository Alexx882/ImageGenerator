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
    provider = MNIST_provider(1000)
    gan = DEGAN_MNIST()

    gan.set_training_data(provider.get_numbers)
    gan.train(epochs=1)
    gan.generate_gif()
    gan.export()

if __name__ == "__main__":
    mnist_training()
    
    

