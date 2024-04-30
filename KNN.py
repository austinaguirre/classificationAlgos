import numpy as np
from LoadClassificationDataSet import LoadClassificationDataSet
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k  # number of clusters
        self.max_iterations = max_iterations  # maximum number of iterations to run the algorithm
        self.centroids = None  # to store the centroids
        self.labels = None

    def fit(self, X):
        # Initialize centroids
        self._init_centroids(X)
        
        for _ in range(self.max_iterations):
            # Assign clusters
            self._assign_clusters(X)
            
            old_centroids = self.centroids.copy()

            # Update centroids
            self._update_centroids(X)
            
            if np.allclose(self.centroids, old_centroids):
                break  # Exit loop if centroids have not changed

            # Optionally, you can implement a convergence check and break the loop if the centroids do not change significantly.

    def _init_centroids(self, X):
        centroids = [X[np.random.choice(len(X))]]  # Choose the first centroid randomly from the data points
        for _ in range(1, self.k):
            distances = np.array([np.min([np.linalg.norm(x - centroid) for centroid in centroids]) for x in X])
            probabilities = distances**2  # Squared distances for the probability distribution
            probabilities /= probabilities.sum()  # Normalize to form a probability distribution
            cumulative_probabilities = np.cumsum(probabilities)
            
            r = np.random.rand()
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(X[i])
                    break
        self.centroids = np.array(centroids)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # Calculate distances from each point to each centroid
        self.labels = np.argmin(distances, axis=1)  # Assign each point to the closest centroid

    def _update_centroids(self, X):
        for i in range(self.k):
            points_in_cluster = X[self.labels == i]
            if points_in_cluster.size > 0:
                self.centroids[i] = np.mean(points_in_cluster, axis=0)  # Update centroid to mean of assigned points

    def _compute_distance(self, X, centroids):
        return np.linalg.norm(X - centroids, axis=1)

    def plot_clusters(self, X):
        plt.figure(figsize=(8, 6))
        colors = plt.cm.get_cmap('viridis', self.k)
        
        for i in range(self.k):
            points_in_cluster = X[self.labels == i]
            centroid = self.centroids[i]
            plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], s=50, c=[colors(i)], label=f'Cluster {i+1}')
            plt.scatter(centroid[0], centroid[1], s=200, c='red', marker='X')  # Mark centroids
        
        plt.title('K-Means Clustering')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()


# X = LoadClassificationDataSet()
# # LoadClassificationDataSetChosen('blobs_classification_dataset.csv')
# kmeans = KMeans(k=5)
# kmeans.fit(X)
# labels = kmeans.labels
# print("Labels:", kmeans.labels)
# kmeans.plot_clusters(X)




