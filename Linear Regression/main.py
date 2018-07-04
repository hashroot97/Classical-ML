import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*3 + 2
y_data = np.vectorize(lambda y_data: y_data + np.random.normal(loc=0.0, scale=0.1))(y_data)
print(x_data.shape, y_data.shape)

# Model

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a*x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init = tf.global_variables_initializer()

# Training

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        _, a_p, b_p = sess.run([optimizer, a, b])
        if i%10 == 0:
            print(i,[a_p, b_p])
        final = [a_p, b_p]

# Plots

[a, b] = final
cr, cg, cb = (0.0, 0.0, 1.0)
f_y = np.vectorize(lambda x: a*x +b)(x_data)
line = plt.plot(x_data, f_y)
plt.setp(line, color=(cr, cg, cb))
plt.plot(x_data, y_data, 'ro')
plt.show()