import sys
import os
from typing import Iterable, List
from PIL import Image
import numpy as np
import random

sys.path.insert(1, '.')
from preprocessing import cropper

class TrainingDataProvider:
    '''This class provides easy and abstracted access to facial image training data.'''

    IMAGE_FILE_ENDINGS = ('.jpg', '.jpeg', '.png')

    def __init__(self, 
                 training_data_locations = ['training_images/lfw-deepfunneled', 'training_images/real_and_fake_face/training_real'],
                 image_width = 28, image_height = 28,
                 batch_size = 256):
        self.training_data_locations = training_data_locations
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

    def _load_from_disk_in_batches(self) -> Iterable[List[str]]:
        '''
        Returns all images in self.training_data_locations in batches of self.batch_size. The order is shuffled every time.
        
        :returns: a generator of lists of paths of images
        '''
        current_batch = [] # the current batch to fill
        current_files = [] # the files to process

        for location in self.training_data_locations:
            for cur_path, _, files in os.walk(location):
                current_files.extend([
                    os.path.normpath(cur_path + '/' + f) 
                    for f in files 
                    if f.endswith(TrainingDataProvider.IMAGE_FILE_ENDINGS)
                ])
        
        random.shuffle(current_files)

        # add only number of files to current batch so it reaches batch size
        while len(current_files) > 0:
            nr_files_to_add = min(len(current_files), self.batch_size - len(current_batch))
            current_batch.extend(current_files[:nr_files_to_add])
            current_files = current_files[nr_files_to_add:]

            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []

        # TODO think about threshold min_size for last batch
        if True: #len(current_batch) > self.batch_size / 4:
            yield current_batch                 

    def get_all_training_images_in_batches(self) -> Iterable[np.ndarray]:
        '''
        Returns all training images in multiple batches, eg. N batches with Y images each. Y is self.batch_size.

        :returns: array with shape (self.batch_size, self.image_width, self.image_height, 1)
        '''

        for image_batch in self._load_from_disk_in_batches():
            current_batch = []
            
            for image_path in image_batch:
                image: Image = Image.open(image_path)
                image = cropper.scale_down(image, self.image_width, self.image_height)
                image = image.convert('L') # FIXME greyscale for now
                
                current_batch.append(np.array(image))

            array = np.asarray(current_batch)
            array = array.reshape(array.shape[0], self.image_width, self.image_height, 1).astype('float32')
            array = (array - 255/2.) / (255/2.) 
                
            yield array

    def store_image_arrays_to_disk(self):
        '''Converts all images as in self.get_all_training_images_in_batches() and stores them in a np file.'''
        arrs = []
        for arr in self.get_all_training_images_in_batches():
            arrs.append(arr)

        np.save('training_images/npdata', np.asarray(arrs))

    def get_all_training_images_in_batches_from_array_on_disk(self) -> Iterable[np.ndarray]:
        '''
        Loads preprocessed image arrays from file.
        This is faster than recalculating the array from the raw image but uses a lot more memory.
        The file is saved with self.store_image_arrays_to_disk().
        
        :returns: same as self.get_all_training_images_in_batches() but faster.
        '''
        arrs = np.load('training_images/npdata.npy', allow_pickle=True)
        random.shuffle(arrs)
        for arr in arrs:
            yield arr


if __name__ == '__main__':
    p = TrainingDataProvider()
    sum = 0
    for batch in p._load_from_disk_in_batches():
        print(len(batch))
        sum += len(batch)
    print(sum)
