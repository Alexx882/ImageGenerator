import sys
import os
from typing import Iterable, List
from PIL import Image
import numpy as np

sys.path.insert(1, '.')
from preprocessing import cropper

class TrainingDataProvider:
    '''This class provides easy and abstracted access to facial image training data.'''

    IMAGE_FILE_ENDING = '.jpg'

    def __init__(self, 
                 training_data_locations = ['training_images/lfw-deepfunneled', 'training_images/real_and_fake_face/training_real'],
                 batch_size = 256):
        self.training_data_locations = training_data_locations
        self.batch_size = batch_size

    def _load_from_disk_in_batches(self) -> Iterable[List[str]]:
        '''
        Returns all images in self.training_data_locations in batches of self.batch_size.
        
        :returns: a generator of lists of paths of images
        '''
        current_batch = [] # the current batch to fill
        current_files = [] # the current files to read in

        for location in self.training_data_locations:
            for cur_path, _, files in os.walk(location):
                # add only number of files to current batch so it reaches batch size
                # add the remainder to next batch (or add empty list)
                current_files = [os.path.normpath(cur_path + '/' + f) for f in files if f.endswith(TrainingDataProvider.IMAGE_FILE_ENDING)]
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
        # TODO fix, currently only returning the complete result of ALL IMAGES!
        Returns all training images in multiple batches, eg. N batches with Y images each. Y is self.batch_size.

        :returns: array with shape (self.batch_size, 28, 28, 1)
        '''
        current_batch = []

        for image_batch in self._load_from_disk_in_batches():

            for image_path in image_batch:
                image: Image = Image.open(image_path)
                image = cropper.scale_down(image, 28, 28)
                image = image.convert('L') # FIXME greyscale for now
                
                current_batch.append(np.array(image))

        array = np.asarray(current_batch)
        array = array.reshape(array.shape[0], 28, 28, 1).astype('float32')
        array = (array - 255/2.) / (255/2.) 
            
        yield array


if __name__ == '__main__':
    p = TrainingDataProvider()
    sum = 0
    for batch in p.get_all_training_images_in_batches():
        print(len(batch))
        sum += len(batch)
    print(sum)
