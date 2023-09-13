import pandas as pd
import numpy as np
import logging
import urllib3
from abc import ABC, abstractmethod
import os


class Dataset(ABC):
    def __init__(self, url: str, save_directory: str, zip_name: str, csv_dataset: str, csv_label_names: str):
        self.url = url
        self.save_directory = save_directory
        self.zip_name = zip_name
        self.csv_dataset = csv_dataset
        self.csv_label_names = csv_label_names

        self.zip_path = None
        self.extracted_path = None
        self.data = None
        self.labels = None
        self.label_names = None
        self.csv_dataset_path = None
        self.csv_names_path = None

        self.dataset_pipeline()

    def download_dataset(self):
        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Download the dataset
        http = urllib3.PoolManager()
        response = http.request(method='GET', url=self.url)
        if response.status == 200:
            self.zip_path = os.path.join(self.save_directory, self.zip_name + '.tar.gz')
            with open(self.zip_path, 'wb') as file:
                file.write(response.data)
            logging.info("Dataset downloaded successfully.")
        else:
            logging.critical("Failed to download the dataset.")
            raise Exception("Failed to download the dataset.")

    def extract_from_zip(self):
        import tarfile
        with tarfile.open(self.zip_path, 'r:gz') as tar:
            tar.extractall(self.save_directory)
            created_directory = tar.getnames()[0].split('/')[0]
        self.extracted_path = os.path.join(self.save_directory, created_directory)

    @abstractmethod
    def load_dataset(self):
        pass

    def convert_to_csv(self):

        csv_data = np.concatenate((self.data, self.labels[:, np.newaxis]), axis=1)
        csv_columns = ['pixel_{}'.format(i) for i in range(self.data.shape[1])] + ['label']

        csv_df = pd.DataFrame(csv_data, columns=csv_columns)
        csv_df_names = pd.DataFrame(np.array(self.label_names), columns=["coarse_label_names"])

        # Create the data folder if it doesn't exist
        self.save_directory = os.path.join(self.save_directory, 'data')
        os.makedirs(self.save_directory, exist_ok=True)

        self.csv_dataset_path = os.path.join(self.save_directory, self.csv_dataset)
        csv_df.to_csv(self.csv_dataset_path, index=False)

        self.csv_names_path = os.path.join(self.save_directory, self.csv_label_names)
        csv_df_names.to_csv(self.csv_names_path, index=False)
        logging.info("Dataset converted to CSV successfully.")

    @abstractmethod
    def dataset_pipeline(self):
        pass
