import os
import tensorflow as tf
import numpy as np
import cv2
from .preprocess_utils import preprocess_input

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=32, n_channels=3, n_classes=3, 
        shuffle=True, data_flag = 'train', dim= None, **augmentation_kwargs):
        
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.data_flag = data_flag
        self.dim = dim
        self.augmentor= tf.keras.preprocessing.image.ImageDataGenerator(**augmentation_kwargs)
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in batch_indexes]


        # Generate data
        X, y= self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype= np.float32)
        y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)
        # Generate data
        for i, idx in enumerate(list_IDs_temp):
            img_path = os.path.join(self.data_dir, idx)
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.dim)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if (self.n_channels == 1):
                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)
            X[i,] = image
            y[i,] = self.labels[idx]
        if(self.data_flag=='train'):
            X, y= self.augmentor.flow(X,y=y,shuffle=False, batch_size=self.batch_size)[0]
        X = preprocess_input(X, v2=True)
        y_final = (y[:,[0]], y[:,[1]], y[:,[2]])
        return X, y_final
