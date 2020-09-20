import sys
import os
from typing import Iterable, List, Any
from PIL import Image
import numpy as np
import random
import shutil

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

        # get absolute path on disc
        self.npy_data_path = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + '/npy/')

    def _get_item_batches(self, items: Iterable[Any], batch_size: int) -> Iterable[Any]:
        current_batch = [] # the current batch to fill

        # add only number of files to current batch so it reaches batch size
        while len(items) > 0:
            nr_files_to_add = batch_size - len(current_batch)
            current_batch.extend(items[:nr_files_to_add])
            items = items[nr_files_to_add:]

            if len(current_batch) == batch_size:
                yield current_batch
                current_batch = []

        # also return last few images as they are different everytime anyway
        if len(current_batch) > 0:
            yield current_batch

    def _load_all_images_from_disk(self) -> List[str]:
        '''
        Returns all images in self.training_data_locations. The order is shuffled every time.
        
        :returns: a list of paths of images
        '''
        current_files = [] # the files to process

        for location in self.training_data_locations:
            for cur_path, _, files in os.walk(location):
                current_files.extend([
                    os.path.normpath(f'{cur_path}/{f}')
                    for f in files 
                    if f.endswith(TrainingDataProvider.IMAGE_FILE_ENDINGS)
                ])
        
        random.shuffle(current_files)

        return current_files
 
    def _preprocess_image(self, image: Image) -> Image:
        '''Preprocesses one image by squarifying it around the face. Throws LookupError if no face was found.'''
        # change to desired size
        image = cropper.scale_down(image, self.image_width, self.image_height)

        # crop around head while keeping desired size
        c = cropper.detect_face(image)
        smaller = min(c[0], c[1])
        c = (c[0]-smaller, c[1]-smaller, self.image_width, self.image_height)
        image = cropper.squarify(image, c)

        # FIXME greyscale for now
        image = image.convert('L')

        return image
            
    def _convert_image_batch_to_training_array(self, images: List[np.ndarray]) -> np.ndarray:
        '''Converts a list of individual image arrays to a single array preprocessed for training.'''
        array = np.asarray(images)
        array = array.reshape(array.shape[0], self.image_width, self.image_height, 1).astype('float32')
        array = (array - 127.5) / (127.5) 
        return array

    def get_all_training_images_in_batches(self) -> Iterable[np.ndarray]:
        '''
        Preprocesses all training images and returns them in multiple batches, where one batch is one numpy array.

        :returns: arrays with shape (self.batch_size, self.image_width, self.image_height, 1)
        '''
        current_batch = [] # the current batch to fill

        for image_path in self._load_all_images_from_disk():
            image: Image = Image.open(image_path)
            try:
                image = self._preprocess_image(image)
            except LookupError:
                continue # no face was found in image

            current_batch.append(np.array(image))

            if len(current_batch) == self.batch_size:
                yield self._convert_image_batch_to_training_array(current_batch)
                current_batch = []
        
        if len(current_batch) > 0:
            yield self._convert_image_batch_to_training_array(current_batch)

    def store_image_arrays_on_disk(self, force=False):
        '''Stores the batches from self.get_all_training_images_in_batches() on the disk as individual array files.'''
        if os.path.exists(self.npy_data_path):
            if force:
                shutil.rmtree(self.npy_data_path)
            else:
                print("Image arrays already exist, aborting.")
                return
            
        if not os.path.exists(self.npy_data_path):
            os.mkdir(self.npy_data_path)

        idx = 0
        for batch in self.get_all_training_images_in_batches():
            np.save(
                os.path.normpath(f'{self.npy_data_path}/{idx}.npy'), 
                batch
            )
            idx += 1

    def get_all_training_images_in_batches_from_disk(self) -> Iterable[np.ndarray]:
        '''
        Loads preprocessed image arrays from files. 
        Memory load is not that high, as only individual batches are read in.
        The batch size is fixed to 64.

        :returns: same as self.get_all_training_images_in_batches() but faster.
        '''
        if not os.path.exists(self.npy_data_path):
            raise IOError(f"The training arrays folder {self.npy_data_path} does not exist.")

        for _, _, files in os.walk(self.npy_data_path):
            random.shuffle(files)
            for file_ in files:
                yield np.load(os.path.join(self.npy_data_path, file_), allow_pickle=True)


if __name__ == '__main__':
    pass
    # store all preprocessed image arrays on disk

    # p = TrainingDataProvider([r'E:\Projects\ImageGenerator\training_images\img_align_celeba_png'], 256, 256, 64)

    # p.store_image_arrays_on_disk()

    # for batch in p.get_all_training_images_in_batches_from_disk():
    #     print(batch.shape)
    #     import matplotlib.pyplot as plt
    #     plt.imshow((batch[0, :,:,0] ), cmap='gray')
    #     plt.show()
    #     break
