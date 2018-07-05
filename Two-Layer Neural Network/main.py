import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
Dim_in = 1000
Dim_out = 10
H = 100
learning_rate = 1e-6
epochs = 300

# Inputs and Outputs
X = np.random.randn(batch_size, Dim_in)
Y_true = np.random.randn(batch_size, Dim_out)

# Weights
W_1 = np.random.randn(Dim_in, H)
W_2 = np.random.randn(H, Dim_out)

# Putting it all together
losses = []
for i in range(epochs+1):
    Dense_OP = np.matmul(X, W_1)
    Relu_OP = np.maximum(Dense_OP, 0)
    Y_pred = np.matmul(Relu_OP, W_2)

    loss = np.square(Y_true - Y_pred).sum()
    if i % 10 == 0 and i >= 50:
        losses.append(loss)
    if i % 50 == 0 and i != 0:
        print('Loss After {} epochs : {}'.format(i, loss))

    grad_Y_pred = 2 * (Y_pred - Y_true)
    grad_W_2 = Relu_OP.T.dot(grad_Y_pred)
    grad_Relu_OP = grad_Y_pred.dot(W_2.T)
    grad_H = grad_Relu_OP.copy()
    grad_H[H < 0] = 0
    grad_W_1 = X.T.dot(grad_H)

    W_1 = W_1 - learning_rate*grad_W_1
    W_2 = W_2 - learning_rate*grad_W_2

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
