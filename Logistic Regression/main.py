import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Dataset

iris = load_iris()
iris_X, iris_y = iris.data[:-1, :], iris.target[:-1]
iris_Y = pd.get_dummies(iris_y).values
iris_X.shape, iris_Y.shape

train_X, test_X, train_Y, test_Y = train_test_split(
    iris_X, iris_Y, test_size=0.33, random_state=42
)
train_X.shape, train_Y.shape, test_X.shape, test_Y.shape

numFeatures = train_X.shape[1]
num_Labels = train_Y.shape[1]

# Model

X = tf.placeholder(tf.float32, [None, numFeatures])
Y_true = tf.placeholder(tf.float32, [None, num_Labels])

weights = tf.Variable(tf.random_normal(
    [numFeatures, num_Labels], mean=0.0, stddev=0.01), name='weights')
biases = tf.Variable(
    tf.random_normal([num_Labels], mean=0.0, stddev=0.01), name='biases'
)

dense_1 = tf.add(tf.matmul(X, weights), biases)
activation_1 = tf.sigmoid(dense_1)
learning_rate = tf.train.exponential_decay(learning_rate=0.0008,
                                           global_step=1,
                                           decay_steps=train_X.shape[0],
                                           decay_rate=0.95,
                                           staircase=True)
loss = tf.nn.l2_loss(activation_1 - Y_true, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(
    tf.argmax(activation_1, axis=1), tf.argmax(Y_true, axis=1)
)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Training

epochs = 1000
acc_arr = []
loss_arr = []
steps_arr = []
with tf.Session() as sess:
    sess.run(init)
    feed_dict = {X: train_X, Y_true: train_Y}
    for i in range(epochs):

        _, ls, acc = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)

        if i % 50 == 0:
            loss_arr.append(ls)
            acc_arr.append(acc)
            steps_arr.append(i)
    loss_arr = loss_arr / loss_arr[np.argmax(loss_arr)]
    print(loss_arr, acc_arr, sep='\n'+100*'-'+'\n')

# Plots

plt.plot(steps_arr, acc_arr, 'g')
plt.plot(steps_arr, loss_arr, 'r')
plt.xlabel('Steps')
plt.ylabel('Accuracy/Loss')
plt.show()
