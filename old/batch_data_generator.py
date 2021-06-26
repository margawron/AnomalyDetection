"""
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import math

class BatchDataGenerator:

    def __init__(self, images, batch_size):
        self.images = images
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        images_batch = self.images[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        return images_batch