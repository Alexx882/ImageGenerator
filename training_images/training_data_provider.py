import os
from typing import Iterable, List

class TrainingDataProvider:
    '''This class provides easy and abstracted access to facial image training data.'''

    IMAGE_FILE_ENDING = '.jpg'

    def __init__(self, batch_size=256):
        self.batch_size = batch_size

    def load_from_disk_in_batches(self, locations: List[str]) -> Iterable[List[str]]:
        '''Returns all locations of images in the list of data locations in batches of self.batch_size.'''
        current_batch = [] # the current batch to fill
        current_files = [] # the current files to read in

        for location in locations:
            for _, _, files in os.walk(location):
                # add only number of files to current batch so it reaches batch size
                # add the remainder to next batch (or add empty list)
                current_files = [f for f in files if f.endswith(TrainingDataProvider.IMAGE_FILE_ENDING)]
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

    def get_all_training_images_in_batches(self) -> Iterable:
        '''Returns all training images in multiple batches, eg. N batches with Y images each. Y is self.batch_size.'''
        for image_data in self.load_from_disk_in_batches():
            current_batch = []

            image: Image = Image.open(io.BytesIO(image_data))
            try:
                image = self.crop_to_face(image)
                current_batch.append(image)
            except AssertionError:
                pass
            
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []


p = TrainingDataProvider()
sum = 0
for batch in  p.load_from_disk_in_batches(['training_images/lfw-deepfunneled', 'training_images/real_and_fake_face/training_real']):
    print(len(batch))
    sum += len(batch)
print(sum)
