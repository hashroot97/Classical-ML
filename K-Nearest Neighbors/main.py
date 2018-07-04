import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def distance_euclidian(p1, p2):
    dist = np.sqrt(((p1-p2)**2).sum())
    return dist

def get_accuracy(kx, x_test, y_test, x_train, y_train):
    preds = []
    for ix in range(x_test.shape[0]):
        label = KNN(x_train, y_train, x_test[ix], k=kx)
        preds.append(label)
    preds = np.array(preds)
    return 100*float((preds==y_test).sum())/y_test.shape[0] 

def KNN(X_Train, Y_Train, X_Test, k=5):
    vals = []
    for  ix in range(X_Train.shape[0]):
        dst = distance_euclidian(X_Train[ix], X_Test)
        vals.append((dst, Y_Train[ix]))
    sorted_vals = sorted(vals, key=lambda mn:mn[0])
    neighbors = np.array(sorted_vals)[:k, -1]
    freq = np.unique(neighbors, return_counts=True)
    my_ans = freq[0][freq[1].argmax()]
    return my_ans

ds = pd.read_csv('./../Datasets/MNIST-Digit/train.csv')
print(ds.shape)
data = ds.values[:5000, :]

split = int(data.shape[0]*0.8)
print(split)
split = int(data.shape[0]*0.95)

X_Train = data[:split, 1:]
Y_Train = data[:split, 0]

X_Test = data[split: , 1:]
Y_Test = data[split:, 0]
print(X_Test.shape, Y_Test.shape, Y_Train.shape, Y_Train.shape)

ans = KNN(X_Train, Y_Train, X_Test[100])

acc = []
for ix in range(3, 9, 2):
    acc = get_accuracy(5, X_Test, Y_Test, X_Train, Y_Train)
    print('Accuracy : {}'.format(acc))
print(ans)