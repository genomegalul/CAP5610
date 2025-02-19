import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading data.
df = pd.read_csv('./Data/HW2_nonlinear_data.csv')
X = df['X'].values
Y = df['Y'].values

# Converting data to NumPy arrays.
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)
n = len(X)

# Initial parameters.
a = 0.0
b = 0.0
c = 0.0
d = 0.0
lr = 1e-6
epochs = 10000

# Gradient descent loop.
for _ in range(epochs):
    # Predicted value.
    Y_pred = a*X**3 + b*X**2 + c*X + d
    
    # Partial derivatives of MSE with respect to a, b, c, and d.
    da = (-2.0 / n) * np.sum(X**3 * (Y - Y_pred))
    db = (-2.0 / n) * np.sum(X**2 * (Y - Y_pred))
    dc = (-2.0 / n) * np.sum(X    * (Y - Y_pred))
    dd = (-2.0 / n) * np.sum((Y - Y_pred))
    
    # Updating.
    a -= lr * da
    b -= lr * db
    c -= lr * dc
    d -= lr * dd

# Printing final parameters.
print(f"Final a (coefficient for X^3): {a}")
print(f"Final b (coefficient for X^2): {b}")
print(f"Final c (coefficient for X^1): {c}")
print(f"Final d (intercept):           {d}")

# Setting up the fit line.
X_line = np.linspace(X.min(), X.max(), 100)
Y_line = a*X_line**3 + b*X_line**2 + c*X_line + d

# Producing the overfitted scatterplot.
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X_line, Y_line, color='red', label='Fitted Cubic')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Nonlinear Regression via Gradient Descent')
plt.savefig('NonlinearRegression.png')
plt.show()
