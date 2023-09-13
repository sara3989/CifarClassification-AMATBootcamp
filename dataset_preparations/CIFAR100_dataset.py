from dataset_preparations.dataset import Dataset

import os
import pickle
import numpy as np


class CIFAR100Dataset(Dataset):

    def __init__(self, url: str, save_directory: str, zip_name: str, csv_dataset: str, csv_label_names: str):
        super().__init__(url, save_directory, zip_name, csv_dataset, csv_label_names)

    def load_dataset(self):
        def unpickle(_file):
            with open(_file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='latin1')
            return data_dict

        file = os.path.join(self.extracted_path, 'train')
        test_file = os.path.join(self.extracted_path, 'test')
        meta_file = os.path.join(self.extracted_path, 'meta')

        # Load training data
        _dict = unpickle(file)
        self.data = _dict['data'].reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        self.labels = np.array(_dict['coarse_labels'])

        # Load test data
        test_dict = unpickle(test_file)
        test_data = test_dict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        test_labels = np.array(test_dict['coarse_labels'])

        # Load meta data
        meta_dict = unpickle(meta_file)
        self.label_names = meta_dict['coarse_label_names']

        self.data = np.concatenate((self.data, test_data), axis=0)
        self.labels = np.concatenate((self.labels, test_labels), axis=0)

        self.data = self.data.reshape(60000, -1)

    def dataset_pipeline(self):
        self.download_dataset()
        self.extract_from_zip()
        self.load_dataset()
        self.convert_to_csv()
