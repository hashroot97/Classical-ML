import pandas as pd
import math
from sklearn.model_selection import train_test_split


def seperateByClass(dataset_X, dataset_Y):
    seperated = {}
    for i in range(dataset_X.shape[0]):
        if dataset_Y[i] not in seperated:
            seperated[dataset_Y[i]] = []
            seperated[dataset_Y[i]].append(dataset_X[i])
        else:
            seperated[dataset_Y[i]].append(dataset_X[i])
    return seperated, list(seperated.keys())


def mean(data):
    return sum(data)/float(len(data))


def stddev(data):
    av = mean(data)
    var = sum([pow(x-av, 2) for x in data])/float(len(data)-1)
    return math.sqrt(var)


def summarizeByClass(dataset):
    summary = {
        keys_seperated[0]: [],
        keys_seperated[1]: []
    }
    for i in keys_seperated:
        for j in range(len(dataset[i][0])):
            arr_fed = [dataset[i][k][j] for k in range(len(dataset[i]))]
            av = mean(arr_fed)
            std = stddev(arr_fed)
            summary[keys_seperated[int(i)]].append([av, std])
    return summary


def pdf(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def get_probability(inputVector):
    probablities = {}
    for j in keys_seperated:
        total_pdf = 1
        for i in range(len(summary[j])):
            m = summary[j][i][0]
            s = summary[j][i][1]
            feat_pdf = pdf(inputVector[i], m, s)
            total_pdf = total_pdf * feat_pdf
        probablities[j] = total_pdf
    return probablities


def get_probability_dataset(data):
    data_probabilities = []
    for i in range(data.shape[0]):
        prob = get_probability(data[i])
        if prob[keys_seperated[0]] > prob[keys_seperated[1]]:
            data_probabilities.append(keys_seperated[0])
        else:
            data_probabilities.append(keys_seperated[1])
    return data_probabilities


def get_accuracy(true_Y, pred_Y):
    count = 0
    for i in range(true_Y.shape[0]):
        if true_Y[i] == pred_Y[i]:
            count += 1
    acc = (float)(count/true_Y.shape[0])
    print("Accuracy : {}".format(acc))


df = pd.read_csv('./../Datasets/diabetes-data.csv', header=None)
df.dropna()
data_X = df.values[:, :-1]
data_Y = df.values[:, -1]

train_X, test_X, train_Y, test_Y = train_test_split(
    data_X, data_Y, test_size=0.33, random_state=42
)
seperated, keys_seperated = seperateByClass(train_X, train_Y)
summary = summarizeByClass(seperated)
pred_Y = get_probability_dataset(test_X)
get_accuracy(test_Y, pred_Y)
