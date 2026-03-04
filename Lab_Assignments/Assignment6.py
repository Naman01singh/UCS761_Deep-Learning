import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.figure()
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Activation")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid()
plt.show()

def tanh(x):
    return np.tanh(x)

plt.figure()
plt.plot(x, tanh(x))
plt.title("Tanh Activation")
plt.xlabel("x")
plt.ylabel("Tanh(x)")
plt.grid()
plt.show()

def relu(x):
    return np.maximum(0, x)

plt.figure()
plt.plot(x, relu(x))
plt.title("ReLU Activation")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid()
plt.show()

def leaky_relu(x, alpha = 0.01):
    return np.where(x > 0, x, alpha * x)

plt.figure()
plt.plot(x, leaky_relu(x))
plt.title("Leaky ReLU Activation")
plt.xlabel("x")
plt.ylabel("LeakyReLU(x)")
plt.grid()
plt.show()

def elu(x, alpha = 1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

plt.figure()
plt.plot(x, elu(x))
plt.title("ELU Activation")
plt.xlabel("x")
plt.ylabel("ELU(x)")
plt.grid()
plt.show()

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
softmax_values = softmax(x)

plt.figure()
plt.plot(x, softmax_values)
plt.title("Softmax Activation")
plt.xlabel("x")
plt.ylabel("Softmax(x)")
plt.grid()
plt.show()

def softplus(x):
    return np.log(1 + np.exp(x))

plt.figure()
plt.plot(x, softplus(x))
plt.title("Softplus Activation")
plt.xlabel("x")
plt.ylabel("Softplus(x)")
plt.grid()
plt.show()

def binary_step(x):
    return np.where(x >= 0, 1, 0)

plt.figure()
plt.plot(x, binary_step(x))
plt.title("Binary Step Activation")
plt.xlabel("x")
plt.ylabel("BinaryStep(x)")
plt.grid()
plt.show()

def piecewise_linear(x):
    return np.clip(x, -2, 2)

plt.figure()
plt.plot(x, piecewise_linear(x))
plt.title("Piecewise Linear Activation")
plt.xlabel("x")
plt.ylabel("PiecewiseLinear(x)")
plt.grid()
plt.show()
