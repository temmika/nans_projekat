import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, labels, predictions=None):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        if predictions is None:
            plt.title(labels[i])
        else:
            plt.title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
