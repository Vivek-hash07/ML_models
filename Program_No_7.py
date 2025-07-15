import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Load Iris dataset and use only 2 features
iris = load_iris()
X = iris.data[:, :2] # Sepal length & width
y = iris.target
# Split data
X_train, X_test, y_train, y_test = train_test_split(X,
y, test_size=0.2, random_state=42)
# Fit KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,
0.02),np.arange(y_min, y_max, 0.02))
# Predict over grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel2,alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,edgecolors='k', label='Train', cmap=plt.cm.Set1)
plt.scatter(X_test[:, 0], X_test[:, 1] ,c=y_test,marker='x', label='Test', cmap=plt.cm.Set1)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("KNN Decision Boundary (k=5)")
plt.legend()
plt.grid(True)
plt.show()