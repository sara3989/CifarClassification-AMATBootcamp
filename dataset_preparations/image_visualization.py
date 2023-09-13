import cv2
import matplotlib.pyplot as plt


# visualize original image vs the rotated image
def visualize_rotated(original, rotated):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(2, 2))

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(original_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title('Rotated')
    plt.imshow(rotated_rgb)
    plt.axis("off")

    plt.show()
