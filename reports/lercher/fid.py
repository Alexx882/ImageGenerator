import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# Execute this to avoid internal tf error
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

######

from numpy import cov, sum, trace, iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = sum((mu1 - mu2)**2.0)
    
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
        
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

######

from skimage.transform import resize
import numpy as np
from PIL import Image

# scale an array of images to a new size
def scale_images(images, new_shape, convert_color=True):
    images_list = []
    
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        
        if convert_color:
            # convert from grey to color
            pil = Image.fromarray(new_image.reshape(299,299).astype('uint8'), 'L')
            pil = pil.convert('RGB')
            new_image = np.array(pil)
                        
        images_list.append(new_image)
        
    return np.asarray(images_list)

######

# add the project's base dir to interpreter paths
import sys
sys.path.insert(1, '../../')
from gan import DCGAN_MNIST
from training_images.training_data_provider import TrainingDataProvider

gan = DCGAN_MNIST()
data_provider = TrainingDataProvider()

######

def get_generated_images(gan) -> 'np.ndarray':
    # generate images
    seed = np.random.normal(0, 1, (NR_IMAGES, 100))
    images1 = gan.generator.predict(seed).astype('float32')
    images1 = images1 * 127.5 + 127.5
    return images1

def get_mnist_images() -> 'np.ndarray':
    # load real images
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    images2 = x_train[:NR_IMAGES, ].reshape(NR_IMAGES, 28, 28, 1).astype('float32')
    return images2

def get_face_images() -> 'np.ndarray':
    images_list = []
    for batch in data_provider.get_all_training_images_in_batches_from_disk():
        for img in batch:
            images_list.append(img)
            
            if len(images_list) == NR_IMAGES:
                return np.asarray(images_list)

### 

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

def calculate_fid_for_dataset(architecture=['dcgan', 'hr_dcgan'], dataset=['mnist', 'faces']):
    gan.import_(path=f"../../gan/models/{dataset}/{architecture}{'_reduced_architecture' if dataset=='faces' else ''}/")
    
    images1 = get_generated_images(gan)
    images2 = get_mnist_images() if dataset == 'mnist' else get_face_images()
    
    assert images1.shape[1] == images2.shape[1] # check if faces or mnist was loaded for both
        
    # resize images
    images1 = scale_images(images1, (299,299,1), convert_color=True)
    if dataset == 'mnist':
        images2 = scale_images(images2, (299,299,1), convert_color=True)
    else:
        # real face training data already has color
        images2 = scale_images(images2, (299,299,3), convert_color=False)
            
    # predict features from pretrained network
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    act1 = model.predict(preprocess_input(images1))
    act2 = model.predict(preprocess_input(images2))

    # calculate fid between images1 and images2
    fid = calculate_fid(act1, act2)
    return fid


NR_IMAGES = 100