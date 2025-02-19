import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading data.
df = pd.read_csv('./Data/HW2_linear_data.csv', header=None, names=['X', 'Y'])
X = df['X'].values
Y = df['Y'].values

# Converting data to NumPy arrays.
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)
n = len(X)

# Initial parameters.
m = 0.0
b = 0.0
lr = 0.0001
epochs = 1000

# Gradient descent loop.
for _ in range(epochs):
    # Predicted value.
    Y_pred = m * X + b

    # Partial derivatives of MSE with respect to m and b.
    dm = (-2.0 / n) * np.sum(X * (Y - Y_pred))
    db = (-2.0 / n) * np.sum(Y - Y_pred)

    # Updating.
    m -= lr * dm
    b -= lr * db

# Printing final parameters.
print(f"Final slope (m): {m}")
print(f"Final intercept (b): {b}")

# Setting up the fit line.
X_line = np.linspace(X.min(), X.max(), 100)
Y_line = m * X_line + b

# Producing the overfitted scatterplot.
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X_line, Y_line, color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression via Gradient Descent')
plt.savefig('LinearRegression.png')
plt.show()