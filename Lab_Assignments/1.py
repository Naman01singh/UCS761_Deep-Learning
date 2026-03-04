import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Acer/Downloads/multiple_linear_regression_dataset.csv")

X = df[['age', 'experience']].values
y = df['income'].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
y_mean = y.mean()
y_std = y.std()
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std
X = np.c_[np.ones(X.shape[0]), X]

learning_rate = 0.001
epochs = 1000
weights = np.zeros(X.shape[1])

for _ in range(epochs):
    for i in range(len(X)):
        y_pred = np.dot(X[i], weights)
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]

y_pred_norm = np.dot(X, weights)
y_pred = y_pred_norm * y_std + y_mean

result_table = pd.DataFrame({
    'Age': df['age'],
    'Experience': df['experience'],
    'Actual_Income': df['income'],
    'Predicted_Income': np.round(y_pred, 2)
})

print("Final Weights (bias, age, experience):")
print(weights)
print("\nResult Table\n")
print(result_table)