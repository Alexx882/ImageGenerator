import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gan
import os

def show_sample(sample):
    plt.imshow(sample)
    plt.colorbar()
    plt.show()

def save_imgs(gan, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))

        gen_imgs = gan.generator.predict(noise)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan/images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == "__main__":
    path = Path("gan/models/mnist")

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    gan = gan.GAN((28,28,1), epochs=10, iterations_discriminator=10, iterations_generator=10, f_save= lambda gan, epoch : save_imgs(gan, epoch))

    model_g = Path("gan/models/mnist/generator.h5")
    model_d = Path("gan/models/mnist/discriminator.h5")

    generator = None
    discriminator = None
    if model_g.exists():
        print("[G] found a saved model, importing ...")
        generator = keras.models.load_model(str(model_g))

    if model_d.exists():
        print("[D] found a saved model, importing ...")
        discriminator = keras.models.load_model(str(model_d))

    if generator != None:
        gan.set_generator(generator)
    
    if discriminator != None:
        gan.set_discriminator(discriminator)
    
    gan.bake_combined()
    
    # labels are not needed as the GAN only distinguishes between "real" and "fake"
    (x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # bring x_train into 4d shape (id, row, col, channel) and normalize values to [0;1]
    x_train = [np.array(sample) for sample in x_train]
    x_train = np.array([sample.reshape(28,28,1) for sample in x_train]) / 255.
    
    gan.set_training_data(x_train)
    gan.doctor()
    gan.train()
    
    gan.export(str(path))