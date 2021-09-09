from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)

x_shape = x_train.shape[1:]

train_groups = {}
for i in range(10):
    train_groups[i] = {}
    train_groups[i]["x"] = x_train[y_train == i]
    train_groups[i]["y"] = y_train[y_train == i]
    
test_groups = {}
for i in range(10):
    test_groups[i] = {}
    test_groups[i]["x"] = x_test[y_test == i]
    test_groups[i]["y"] = y_test[y_test == i]
    
label_names = [i for i in range(10)]

x_shape = test_groups[0]["x"][0].shape