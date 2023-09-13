def test_convert_and_load():
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    csv_dataset_path = '../../data/merged_dataset.csv'
    df_dataset = pd.read_csv(csv_dataset_path)
    df_pixels = df_dataset[filter(lambda x: x.startswith('pixel') is True, df_dataset.columns)]

    fig, ax = plt.subplots(figsize=(3, 3), nrows=3, ncols=3)

    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(np.array(df_pixels.iloc[random.choice(range(df_pixels.shape[0]))]).reshape((32, 32, 3)))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()
