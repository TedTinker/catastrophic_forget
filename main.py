import os
from torchinfo import summary

file = r"C:\Users\tedjt\Desktop\Thinkster\87_catastrophic_forget\code"
os.chdir(file)

from get_data import train_groups, test_groups, x_shape
from show_images import show_image
from model import classifier
from train_test import print_rates, session



print()
for key, value in train_groups.items():
    print("{}:\t\t{} images, \t{} labels.".format(key, value["x"].shape, value["y"].shape))
print()
for key, value in test_groups.items():
    print("Test {}:\t{} images, \t{} labels.".format(key, value["x"].shape, value["y"].shape))
print()

show_image(10, 1, True, predicted = 2)

print()
print(classifier)
print()
print(summary(classifier, (1,) + x_shape))



print("\nBefore any training:")
print_rates(classifier)

def normal_training(classifier):
    classifier = session(classifier, [0,1,2,3,4,5,6,7,8,9])
    print("\nAfter training on all labels:")
    print_rates(classifier)

def catastrophic_training(classifier):
    classifier = session(classifier, [0, 1, 2, 3, 4])
    print("\nAfter training on 0-4:")
    print_rates(classifier)
    
    classifier = session(classifier, [5, 6 , 7, 8, 9])
    print("\nAfter training on 5-9:")
    print_rates(classifier)
    
#normal_training(classifier)
catastrophic_training(classifier)