from dataset_preparations.CIFAR10_dataset import CIFAR10Dataset
from dataset_preparations.CIFAR100_dataset import CIFAR100Dataset
from dataset_preparations.merge_CIFAR10_and_100_datasets import MergeCIFAR10And100Datasets
from dataset_preparations.data_augmentation import balance_data


def dataset_preparations():
    # constants
    url_10 = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    url_100 = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    save_directory = "./"
    zip_name_10 = 'CIFAR-10_dataset'
    zip_name_100 = 'CIFAR-100_dataset'
    csv_dataset_10 = 'CIFAR-10_dataset.csv'
    csv_dataset_100 = 'CIFAR-100_dataset.csv'
    csv_names_10 = 'CIFAR-10_names.csv'
    csv_names_100 = 'CIFAR-100_names.csv'

    our_label_names = ['people', 'flowers', 'trees']
    csv_merged_dataset = 'merged_dataset.csv'
    csv_our_label_names = 'dataset_names.csv'
    csv_rotated_10_name = 'rotated_CIFAR-10.csv'
    csv_rotated_100_name = 'rotated_CIFAR-100.csv'

    cifar10 = CIFAR10Dataset(url_10, save_directory, zip_name_10, csv_dataset_10, csv_names_10)
    cifar100 = CIFAR100Dataset(url_100, save_directory, zip_name_100, csv_dataset_100, csv_names_100)

    merge_datasets = MergeCIFAR10And100Datasets(cifar10, cifar100, our_label_names, csv_merged_dataset,
                                                csv_our_label_names)

    df_merged_dataset = merge_datasets.df_merged_dataset

    balance_data(df_merged_dataset, save_directory, csv_rotated_10_name, csv_rotated_100_name)
