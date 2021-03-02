
## Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
## https://github.com/shervinea/enzynet
"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/utils
@author: Franziska Schirrmacher
"""

import numpy as np
import keras

# data generator to load the data during training and perform augmentation
class DataGenerator(keras.utils.Sequence):
    #Initialization of the data generator
    def __init__(self,dataset, training,model_type ):        # size of the output image


        self.dataset = dataset
        self.n_batch = training["n_batch"]
        # indexes that we shuffle during training
        self.indexes = np.arange(len(self.dataset.IDs["train"]))
        self.shuffle = training["shuffle"]

        if model_type == "sr":
            self.loader = self.dataset.load_data_generator_sr
        elif model_type == "cl":
            self.loader = self.dataset.load_data_generator_cl
        else:
            self.loader = self.dataset.load_data_generator_sr2

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset.IDs["train"])/ self.n_batch))

    def __getitem__(self, index):
        # Generate indexes of the batch

        indexes = self.indexes[index*self.n_batch:(index+1)*self.n_batch]
        # Find list of IDs
        list_IDs_temp = [self.dataset.IDs["train"][indexes[i]] for i in range(len(indexes))]
        # Generate data
        return self.loader(list_IDs_temp,self.n_batch)



    def on_epoch_end(self):
        #Updates indexes after each epoch' -> perform shuffling if set to true
        #
        self.indexes = np.arange(len(self.dataset.IDs["train"]))

        if self.shuffle:
            np.random.shuffle(self.indexes)



