from keras.utils import Sequence
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import math
import random
import numpy as np
import os

def load_image(im):
    return img_to_array(load_img(im, color_mode='grayscale')) / 255.

class mySequenceGenerator(Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, data_path, batch_size, mode='train'):
        self.df = df
        self.bsz = batch_size
        self.mode = mode

        # Take labels and a list of image locations in memory
        self.labels = self.df.values[:, 2:]
        self.im_list_0 = self.df['FrameID'].apply(lambda x: os.path.join(data_path, "Images/" + str(x) + "_0.jpg")).tolist()
        self.im_list_1 = self.df['FrameID'].apply(lambda x: os.path.join(data_path, "Images/" + str(x) + "_1.jpg")).tolist()


    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.labels))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        in_0 = np.array([load_image(im) for im in self.im_list_0[idx * self.bsz: (1 + idx) * self.bsz]])
        in_1 = np.array([load_image(im) for im in self.im_list_1[idx * self.bsz: (1 + idx) * self.bsz]])
        return [in_0, in_1]

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y