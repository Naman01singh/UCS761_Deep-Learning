import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/Acer/Downloads/glass.csv")

X = df.drop("Type", axis = 1).values
y = df["Type"].values

y = np.where(y > 1, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.01
epochs = 1000

for _ in range(epochs):
    linear_output = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_output)
    dw = np.dot(X.T, (y_pred - y)) / len(y)
    db = np.sum(y_pred - y) / len(y)
    weights -= learning_rate * dw
    bias -= learning_rate * db

y_final = np.where(y_pred >= 0.5, 1, 0)

accuracy = np.mean(y_final == y)
print("Accuracy:", accuracy * 100)