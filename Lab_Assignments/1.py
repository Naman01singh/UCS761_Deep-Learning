import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

data = pd.read_csv("C:/Users/Acer/Downloads/abalone.data", header = None)

data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight','Viscera_weight', 'Shell_weight', 'Rings']

data['Age'] = data['Rings'] + 1.5
data = pd.get_dummies(data, columns = ['Sex']) 

X = data.drop(['Rings', 'Age'], axis = 1).values
y = data['Age'].values.reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

np.random.seed(42)
input_size = X_train.shape[1]
hidden1_size = 32
hidden2_size = 16
output_size = 1
learning_rate = 0.01
epochs = 1000

W1 = np.random.randn(input_size, hidden1_size) * 0.01
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size) * 0.01
b3 = np.zeros((1, output_size))

for epoch in range(epochs):

    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    output = Z3   

    loss = np.mean((y_train - output) ** 2)

    dZ3 = output - y_train
    dW3 = np.dot(A2.T, dZ3) / X_train.shape[0]
    db3 = np.sum(dZ3, axis = 0, keepdims = True) / X_train.shape[0]

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / X_train.shape[0]
    db2 = np.sum(dZ2, axis = 0, keepdims = True) / X_train.shape[0]

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_train.T, dZ1) / X_train.shape[0]
    db1 = np.sum(dZ1, axis = 0, keepdims = True) / X_train.shape[0]

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = relu(Z2_test)

Z3_test = np.dot(A2_test, W3) + b3
predictions = Z3_test

mse = np.mean((y_test - predictions) ** 2)
rmse = np.sqrt(mse)
print("\nFinal Test MSE:", mse)
print("Final Test RMSE:", rmse)

print("\nSample Predictions vs Actual:\n")
for i in range(10):
    print(f"Predicted: {predictions[i][0]:.2f}  |  Actual: {y_test[i][0]}")