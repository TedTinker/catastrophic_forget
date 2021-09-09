### How to show one of the data

from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from get_data import train_groups, test_groups, label_names

def show_image(index, label, test = False, predicted = None):
    title = "Index {}".format(index)
    if(test):
        title += " in test"
        image = test_groups[label]["x"][index][:,:,:]
    else:
        image = train_groups[label]["y"][index][:,:,:]
    title += "\n(actually {}".format(label)
    if(predicted != None):
        title += ", predicted {}".format(predicted)
    title += ")"
    image = np.transpose(image, (1, 2, 0))
    
    plt.imshow(image, cmap = "gray")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()
    


### How to show training of a model

import datetime

start = datetime.datetime.now()

def duration():
    current = datetime.datetime.now()
    d = current - start
    d = str(d).split(".")[0]
    return(d)    

def show_train(correct_list):
    title = "Test-performance at epoch {}\nDuration: {}".format(len(correct_list) + 1, duration())
    indexes = [i for i in range(1, len(correct_list) + 1)]
    
    plt.plot(indexes, correct_list)
    plt.plot(indexes, [max(correct_list)] * len(correct_list))
    plt.title(title)
    plt.ylim(0, 1)
    plt.show()
    plt.close()