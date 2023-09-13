from dataset_preparations import dataset

import os
import pickle
import numpy as np


class CIFAR10Dataset(dataset.Dataset):
    def __init__(self, url: str, save_directory: str, zip_name: str, csv_dataset: str, csv_label_names: str):
        super().__init__(url, save_directory, zip_name, csv_dataset, csv_label_names)

    def load_dataset(self):
        # Load label names from metadata
        label_names_file = os.path.join(self.extracted_path, 'batches.meta')
        with open(label_names_file, 'rb') as file:
            label_names_dict = pickle.load(file, encoding='bytes')
            self.label_names = [label.decode('utf-8') for label in label_names_dict[b'label_names']]

        self.data = []
        self.labels = []

        # Load data batches
        for i in range(1, 6):
            batch_file = os.path.join(self.extracted_path, 'data_batch_{}'.format(i))
            with open(batch_file, 'rb') as file:
                data_dict = pickle.load(file, encoding='latin1')
            self.data.append(data_dict['data'])
            self.labels.extend(data_dict['labels'])

        # load test batch
        batch_file = os.path.join(self.extracted_path, 'test_batch')
        with open(batch_file, 'rb') as file:
            data_dict = pickle.load(file, encoding='latin1')
        self.data.append(data_dict['data'])
        self.labels.extend(data_dict['labels'])

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels)

        self.data = self.data.reshape((60000, 3, 32, 32)).transpose(0, 2, 3, 1).astype("uint8")
        self.labels = np.array(self.labels)

        self.data = self.data.reshape(60000, -1)

    def dataset_pipeline(self):
        self.download_dataset()
        self.extract_from_zip()
        self.load_dataset()
        self.convert_to_csv()
