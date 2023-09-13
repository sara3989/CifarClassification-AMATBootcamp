from dataset_preparations.image_rotation import rotate_half_of_CIFAR_10, rotate_CIFAR_10
import pandas as pd


def show_images_amount_per_class(df_dataset: pd.DataFrame) -> None:
    """
    Analyze merged dataset - how many images in each class after the merge
    :param df_dataset:
    :return:
    """
    df_dataset['label'].value_counts().plot.bar()


def balance_data(df_dataset: pd.DataFrame, save_directory: str, rotated10_path: str, rotated100_path: str):
    """
    :param df_dataset: DataFrame of dataset
    :param rotated10_path:
    :param rotated100_path:
    :param save_directory:
    :return:
    """
    print(df_dataset)
    sources = df_dataset['source'].unique()

    df_cifar10 = df_dataset[df_dataset['source'] == sources[0]]
    df_cifar100 = df_dataset[df_dataset['source'] == sources[1]]

    df_cifar10_rotated = rotate_half_of_CIFAR_10(df_cifar10)
    df_cifar10_rotated.to_csv(save_directory + '/' + rotated10_path)

    # df_cifar100_rotated = rotate_CIFAR_10(df_cifar100)
    # df_cifar10_rotated.to_csv(save_directory + '/' + rotated100_path)

