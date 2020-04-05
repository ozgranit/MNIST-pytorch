import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import mnist_1
import mnist_2
import mnist_3


def main():
    trainLoss1, accuracy1 = mnist_1.main()
    trainLoss2, accuracy2 = mnist_2.main()
    trainLoss3, accuracy3 = mnist_3.main()
    plt.plot(trainLoss1, marker='o', color='b', label='Original')
    plt.plot(trainLoss2, marker='o', color='y', label='Improved')
    plt.plot(trainLoss3, marker='o', color='m', label='ReLU')
    plt.title("Training Loss of Different Models")
    plt.xlabel('#Epoch')
    plt.ylabel('Training Loss')
    # plt.xlim(0, num_epochs)
    plt.legend()
    plt.show()
    # plot accuracy
    plt.clf()
    accuracy = [accuracy1, accuracy2, accuracy3]
    print(accuracy)
    bars = ('Original', 'Improved', 'ReLU')
    y_pos = np.arange(len(bars))
    # Create bars and choose color
    plt.bar(y_pos, accuracy, width=.5, color=(0.2, 0.7, 0.9))
    # Add title and axis names
    plt.title('Accuracy On The Test Set')
    plt.xlabel('Model')
    plt.ylabel('Accuracy %')
    # Limits for the Y axis
    plt.ylim(0, 100)
    # Create names
    plt.xticks(y_pos, bars)

    for i, v in enumerate(accuracy):
        plt.text(y_pos[i] - 0.1, v + 0.01, str(v) + ' %')
    plt.show()


if __name__ == "__main__":
    main()
