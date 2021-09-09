from random import shuffle
from tqdm import tqdm
import numpy as np
from torch import nn
import torch
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from get_data import train_groups, test_groups, label_names
from model import classifier, opti
from show_images import show_train

loser = nn.CrossEntropyLoss()

def get_batch(label, batch_size):
    batch_index = [i for i in range(train_groups[label]["x"].shape[0])]
    shuffle(batch_index)
    batch_index = batch_index[:batch_size]
    x = train_groups[label]["x"][batch_index]
    y = train_groups[label]["y"][batch_index]
    return(torch.tensor(x).to(device), torch.tensor(y).to(device))

def train(label_list, batch_size):
    opti.zero_grad()
    loss = 0
    for label in label_list:
        x, y = get_batch(label, batch_size // len(label_list))
        y = y.to(device).long()
        y_pred = classifier(x)
        loss += loser(y_pred, y)
    loss.backward()
    opti.step()
    
def train_times(label_list, batch_size = 256, repeat = 1):
    classifier.train()
    for _ in range(repeat):
        train(label_list, batch_size)
    
def get_correct(x, y, correct, total):
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    xs = torch.split(x, 100)
    ys = torch.split(y, 100)
    for x_part, y_part in zip(xs, ys):
        y_pred = classifier(x_part.to(device))
        y_pred = torch.argmax(y_pred, dim = 1)
        for real_y, pred_y in zip(y_part, y_pred):
            total += 1
            if(real_y == pred_y):
                correct += 1
    return(correct, total)
    
def test(label_list):
    classifier.eval()
    correct, total = 0, 0
    for label in label_list:
        x = test_groups[label]["x"]
        y = test_groups[label]["y"]
        correct, total = get_correct(x, y, correct, total)
    return(correct / total)

def session(label_list, target = .9, length = 500):
    global opti
    opti = Adam(classifier.parameters())
    correct_list = []
    for i in tqdm(range(length)):
        train_times(label_list)
        correct_list.append(test(label_list))
        if(correct_list[-1] > target or i == length - 1):
            print()
            show_train(correct_list)
            break
        
### Show correct/incorrect images

from itertools import product
import matplotlib.pyplot as plt

def get_image_list(actual, predicted):
    classifier.eval()
    x = test_groups[label_names[actual]]["x"]
    y_pred = classifier(torch.tensor(x))
    y_pred = torch.argmax(y_pred, dim = 1)
    image_list = [i for i in range(len(test_groups[label_names[actual]]["y"])) if(y_pred[i] == predicted)]
    return(image_list)

def print_rates():
    all_combos = [j for j in product([i for i in range(10)], [i for i in range(10)])]
    all_image_list = {}
    
    for (actual, predicted) in all_combos:
        image_list = get_image_list(actual,predicted)
        all_image_list[(actual, predicted)] = image_list
        #print("\n{} pictures actually {}, predicted to be {}.".format(len(image_list), label_names[actual], label_names[predicted]))
        #for index in image_list[:1]:
        #    show_image(index, label_names[actual], True, label_names[predicted])
    
    total = sum([len(all_image_list[key]) for key in all_image_list.keys()])
    rates = np.zeros((10,10))
    for key in all_image_list.keys():
        rates[key[1], key[0]] = len(all_image_list[key]) / total
                
    fig, ax = plt.subplots()
    ax.imshow(rates)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    
    plt.show()