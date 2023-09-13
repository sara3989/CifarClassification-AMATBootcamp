import numpy as np
import pandas as pd
import random
import cv2
import logging
from datetime import datetime


def rotate_image(image_array: np.array, angle: float) -> np.array:
    """
    Rotate an image in form of numpy array
    :param image_array: Image to rotate of type numpy array
    :param angle: The angle of rotation
    :return: The rotated image of type numpy array.
    """
    image = image_array.copy()

    height, width = image.shape[:2]
    diagonal = np.sqrt(height ** 2 + width ** 2)
    padding = int((diagonal - min(height, width)) / 2)

    # Add padding to the image using BORDER_REFLECT or BORDER_REPLICATE mode
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    # Calculate the center of the padded image
    center_x = width // 2 + padding
    center_y = height // 2 + padding

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Apply the rotation to the padded image
    rotated = cv2.warpAffine(padded_image, rotation_matrix, (padded_image.shape[1], padded_image.shape[0]),
                             flags=cv2.INTER_LINEAR)

    # Crop the rotated image to remove the padding
    rotated_cropped = rotated[padding:-padding, padding:-padding]

    return rotated_cropped


def rotate_half_of_CIFAR_10(df_cifar10: pd.DataFrame) -> pd.DataFrame:

    rotated_cifar10 = pd.DataFrame()

    labels = df_cifar10['label'].unique()
    pixels = [col for col in df_cifar10.columns if col.startswith('pixel')]

    for label in labels:
        logging.info(f'LABEL: {label}')

        # create a DataFrame of the current label class
        df_class = df_cifar10[df_cifar10['label'] == label]
        # take only half class
        half_class_to_rotate = df_class.iloc[:df_class.shape[0] // 2]
        half_class_to_save = df_class.iloc[df_class.shape[0] // 2:]
        rotated_cifar10 = pd.concat([rotated_cifar10, half_class_to_save])

        for i in range(half_class_to_rotate.shape[0]):

            # Image pixels of current index
            image_pixels = half_class_to_rotate[pixels].iloc[i]

            # Convert into unsigned kind.
            image_array = image_pixels.values.reshape(32, 32, 3).astype('uint8')

            # Create a random angle until 90 degrees
            angle = random.random() * 90

            now = datetime.now()
            logging.info(f"i = {i} -> rotate_image(image_array, {angle}), time: {now}")
            print(now)

            rotated_image = rotate_image(image_array, angle)

            now = datetime.now()
            print(now)

            rotated_image_df = pd.DataFrame(rotated_image.reshape(1, -1), columns=pixels)
            rotated_image_df['label'] = label
            rotated_image_df['source'] = 'cifar-10 rotated'
            rotated_cifar10 = pd.concat([rotated_cifar10, rotated_image_df])

    return rotated_cifar10


def rotate_CIFAR_10(df_cifar100: pd.DataFrame):
    pass
