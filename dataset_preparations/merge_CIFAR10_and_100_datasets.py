import numpy as np

from dataset_preparations import CIFAR10_dataset, CIFAR100_dataset

import pandas as pd
import os


class MergeCIFAR10And100Datasets:
    def __init__(self, cifar10: CIFAR10_dataset.CIFAR10Dataset, cifar100: CIFAR100_dataset.CIFAR100Dataset,
                 our_label_names: list,
                 csv_merge_dataset: str, csv_our_label_names: str):
        self.cifar10 = cifar10
        self.cifar100 = cifar100
        self.our_label_names = our_label_names
        self.csv_merge_dataset = csv_merge_dataset
        self.csv_our_label_names = csv_our_label_names

        self.csv_merge_dataset_path = None
        self.csv_our_label_names_path = None
        self.df_merged_dataset = None

        self.merge_CIFAR_100_to_CIFAR_10()

    def super_class_from_CIFAR_100(self, label_name: str) -> (pd.DataFrame, int):
        """
        :param label_name: label name of desired class from CIFAR-100 dataset.
        :return: DataFrame of the images of the desired label from CIFAR-100 dataset.
        """
        # Read labels names
        df_labels_names = pd.read_csv(self.cifar100.csv_names_path)

        # Read CIFAR-100 dataset
        df_cifar_100 = pd.read_csv(self.cifar100.csv_dataset_path)

        # Encoding label name
        label = df_labels_names.index[df_labels_names['coarse_label_names'] == label_name].tolist()[0]

        # Create a DataFrame of the label class
        df = pd.DataFrame(df_cifar_100.loc[df_cifar_100['label'] == label])

        return df, label

    def super_classes_from_CIFAR_100(self, from_label: int) -> pd.DataFrame:
        """
        :param from_label: first label to encoding the new class label.
        (Respectively to CIFAR-10 labels, i.e. from_label = CIFAR-10 last label + 1)
        :return: DataFrame contains the images of the desired labels from CIFAR-100 dataset.
        """
        # Read CIFAR-10 labels names
        df_cifar10_names = pd.read_csv(self.cifar10.csv_names_path)

        df_classes = pd.DataFrame()
        df_names = pd.DataFrame()

        # last label to encoding the new class label
        to_label = from_label + len(self.our_label_names)

        for label_name, label_index in zip(self.our_label_names, range(from_label, to_label)):

            # df_class will contain images of the current label_name class
            # label will contain the encoding origin label
            df_class, label = self.super_class_from_CIFAR_100(label_name)

            # replace CIFAR-100 origin label to a new proper label
            df_class['label'] = df_class['label'].replace(label, label_index)

            # Concatenate the current class to all classes
            df_classes = pd.concat([df_classes, df_class])

            # Create DataFrame of the new label and concatenate to all labels
            df_new_label_name = pd.DataFrame([label_name], index=[label_index], columns=df_cifar10_names.columns)
            df_names = pd.concat([df_names, df_new_label_name])

            # store the new labels
            df_new_label_name = pd.DataFrame([label_name], index=[label_index], columns=df_names.columns)
            df_names = pd.concat([df_names, df_new_label_name])

        # return a single dataframe object of all required labels images from CIFAR-100
        return df_classes

    def merge_CIFAR_100_to_CIFAR_10(self):

        # Read CIFAR-10 dataset
        df_cifar_10 = pd.read_csv(self.cifar10.csv_dataset_path)

        # Desired classes from CIFAR-100 dataset
        from_label = len(np.unique(self.cifar10.labels))
        df_super_classes = self.super_classes_from_CIFAR_100(from_label)

        # Add column source which contains the origin of data
        df_cifar_10["source"] = "cifar-10"
        df_super_classes["source"] = "cifar-100"

        # concatenate
        self.df_merged_dataset = pd.concat([df_cifar_10, df_super_classes])

        # shuffle df
        # df_cifar_10.sample(frac=1)

        # Save the merged dataframe of CIFAR-10 with required labels images of CIFAR-100
        self.csv_merge_dataset_path = os.path.join(self.cifar10.save_directory, self.csv_merge_dataset)
        self.df_merged_dataset.to_csv(self.csv_merge_dataset_path)
