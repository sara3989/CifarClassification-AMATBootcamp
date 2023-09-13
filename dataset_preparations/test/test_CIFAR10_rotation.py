import pandas as pd

from dataset_preparations import image_visualization


def test_cifar10_rotation():
    df_rotated_cifar10 = pd.read_csv('../data/rotated_CIFAR-10.csv')
    df_cifar10 = pd.read_csv('../data/CIFAR-10_dataset.csv')
    class_size = 6000
    img_index = 3000
    labels = df_cifar10['label'].unique()
    pixels = [col for col in df_cifar10.columns if col.startswith('pixel')]

    # only pixels of images (without label and source etc. columns)
    df_rotated_cifar10 = df_rotated_cifar10[pixels]

    # for each label test rotation by visualization
    for i, label in enumerate(labels):
        # origin CIFAR-10 class of current label
        df_class = df_cifar10[df_cifar10['label'] == label]

        # only pixels of image (without label and source etc. columns)
        df_class = df_class[pixels]

        # first image of current class in the origin CIFAR-10
        origin_img = df_class.iloc[0:1].values.reshape(32, 32, 3).astype('uint8')

        # the corresponded image in the rotated version
        rotated_img = df_rotated_cifar10.iloc[
                      (img_index + class_size * i):(img_index + class_size * i + 1)].values.reshape(
            32, 32, 3).astype('uint8')

        image_visualization.visualize_rotated(origin_img, rotated_img)
