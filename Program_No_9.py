import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X,_ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=30, cmap='gray')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

K_means = KMeans(n_clusters=4,random_state=0)
K_means.fit(X)
y_kmeans = K_means.predict(X)
centers = K_means.cluster_centers_

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200,alpha=0.75, label='Centroids',marker='X')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()