import os
import numpy as np
import pandas as pd
from glob import glob
from base.base_data_loader import BaseDataLoader

class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)

        self.trainA_filenames = np.array(glob('./data/%s/%s/*' % (config['dataset_name'], 'trainA')))
        self.trainB_filenames = np.array(glob('./data/%s/%s/*' % (config['dataset_name'], 'trainB')))
        self.testA_filenames = np.array(glob('./data/%s/%s/*' % (config['dataset_name'], 'testA')))
        self.testB_filenames = np.array(glob('./data/%s/%s/*' % (config['dataset_name'], 'testB')))


    def get_trainA_data(self):
        return self.trainA_filenames

    def get_trainB_data(self):
        return self.trainB_filenames

    def get_testB_data(self):
        return self.testB_filenames        

    def get_testA_data(self):
        return self.testA_filenames








    