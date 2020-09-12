import sys
import os
from PIL import Image
import cv2
import io
import numpy as np
from typing import Iterable
import shutil

sys.path.insert(1, '.')
from crawler import downloader
from preprocessing import cropper

class ImagePreparationPipeline:

    sources_location = 'crawler/sources.txt'
    training_image_location = 'training_images/'

    def __init__(self, batch_size=256):
        self.batch_size = batch_size

    def crawl_image_sources(self, force_update=False):
        '''
        Crawls images from a website and stores all image urls in a single txt file.
        
        :param force_update: force the crawl even if sources exist already
        '''

        if os.path.exists(ImagePreparationPipeline.sources_location) and not force_update:
            print('Sources are already available, not updating (use force_update=True to recrawl).')
            return

        # keep import here as crawler depends on selenium 
        # and workstations without plugin should still execute the rest
        from crawler import crawler, purify_sources

        # load urls of images into local file and removes duplicates
        crawler.run()
        purify_sources.purify()

    def crop_to_face(self, image:Image) -> Image:
        '''Crops the image to only contain the face. Throws an AssertionError if no distinct face was found.'''
        c = cropper.detect_face(image)
        image = cropper.squarify(image, c)
        image = cropper.scale_down(image)
        # check if the face is still there
        cropper.detect_face(image) 
        
        return image

    def store_all_processed_training_images(self):
        '''Stores all training images at some internal location'''
        if os.path.exists(self.training_image_location):
            shutil.rmtree(self.training_image_location)
        if not os.path.exists(self.training_image_location):
            os.mkdir(self.training_image_location)

        cnt: int = 0
        for image_data in downloader.retrieve_images(print_status=True):
            image: Image = Image.open(io.BytesIO(image_data))

            try:
                image = self.crop_to_face(image)
            except AssertionError:
                continue
            
            image.save(os.path.normpath(ImagePreparationPipeline.training_image_location + f'/{cnt}.png'))
            cnt += 1

    # TODO use from disk, as the download and cropping would happen in each epoch otherwise...
    def get_all_training_images_in_batches(self) -> Iterable:
        '''Returns all images in multiple batches, eg. N batches with Y images each.'''

        current_batch = []
        for image_data in downloader.retrieve_images(print_status=True):
            image: Image = Image.open(io.BytesIO(image_data))
            try:
                image = self.crop_to_face(image)
                current_batch.append(image)
            except AssertionError:
                pass
            
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []


if __name__ == '__main__':
    p = ImagePreparationPipeline()
    p.store_all_processed_training_images()
    