import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X = data.data[:, [0]]
y = data.target

model = LinearRegression()

model.fit(X, y)

#predict line
x_range = np.linspace(X.min(),X.max() , 100).reshape(-1, 1)
y_pred = model.predict(x_range)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data',color='Skyblue')
plt.plot(x_range, y_pred, color='red', label='Best Fit Line')
plt.title('Linear Regression : Median Income vs House Age')
plt.xlabel('Median Income')
plt.ylabel('House Value')
plt.legend()
plt.grid(True)
plt.show()