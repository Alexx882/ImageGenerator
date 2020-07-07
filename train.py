import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from gan import GAN, HR_DCGAN
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

ARCHITECTURE_NAME = 'hr_dcgan'

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
        fig.savefig(f"gan/results_{ARCHITECTURE_NAME}/epoch_{epoch}.png")
        plt.close()


def load_dataset():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train =  np.append(x_train, x_test, axis=0)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) / 255.
    # x_train = x_train[:500,]
    # x_train.shape
    return x_train


def train(x_train):
    gan = HR_DCGAN(shape=(28,28,1), f_save=lambda gan, n_epoch: save_imgs(gan, n_epoch))
    gan.import_(f'gan/models_{ARCHITECTURE_NAME}/', silent=True)

    gan.set_training_data(x_train)
    gan.train(epochs=20, iterations_generator=40, iterations_discriminator=40)

    gan.export(f'gan/models_{ARCHITECTURE_NAME}/')


def create_folder(folder, delete_first=False):
    if os.path.exists(folder) and delete_first:
        shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)


create_folder(f'gan/results_{ARCHITECTURE_NAME}/', delete_first=True)
create_folder(f'gan/models_{ARCHITECTURE_NAME}/')

data = load_dataset()
train(data)
