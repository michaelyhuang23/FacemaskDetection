import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ModelCreator import FullProposer
import time
import sys
from DataReader import *

train_dataset = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt', 0.3)

for data in train_dataset[:5]:
    img, *labels = data
    plt.figure()
    plt.imshow(img)
    for i in range(5):
        plt.figure()
        plt.imshow(labels[i])
    plt.show()