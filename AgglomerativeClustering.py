import numpy as np
from LoadClassificationDataSet import LoadClassificationDataSet
import matplotlib.pyplot as plt


class AgglomerativeClustering:
    def __init__(self, data, n_clusters=2, linkage='single'):
        """
        Initializes the AgglomerativeClustering.

        :param data: numpy array of shape (n_samples, n_features)
        :param n_clusters: the number of clusters to find
        :param linkage: the linkage criterion to use ('single', 'complete', 'average')
        """
        self.data = data
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = np.zeros(data.shape[0], dtype=int)  # Cluster labels for each point
        self.distances_ = self._compute_distances()
        self.clusters_ = {i: [i] for i in range(data.shape[0])}  # Dictionary to hold the clusters

    def _compute_distances(self):
        """
        Computes the distance matrix for the dataset.

        :return: Distance matrix as a numpy array
        """
        n_samples = self.data.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.linalg.norm(self.data[i] - self.data[j])
                distances[j, i] = distances[i, j]  # since the distance matrix is symmetric
        return distances

    def fit(self):
        """
        Perform the clustering.
        """
        while len(self.clusters_) > self.n_clusters:
            self._merge_clusters()

    def _merge_clusters(self):
        """
        Merges the closest pair of clusters based on the linkage criteria.
        """
        # Step 1: Identify the closest pair of clusters
        min_distance = np.inf
        to_merge = (None, None)

        for i in self.clusters_:
            for j in self.clusters_:
                if i != j and self.distances_[i, j] < min_distance:
                    min_distance = self.distances_[i, j]
                    to_merge = (i, j)

        # Step 2: Merge the clusters
        cluster1, cluster2 = to_merge
        self.clusters_[cluster1].extend(self.clusters_[cluster2])
        del self.clusters_[cluster2]

        # Step 3: Update the distances matrix for the new cluster
        for k in self.clusters_:
            if k != cluster1:
                if self.linkage == 'single':
                    new_distance = min(self.distances_[cluster1, k], self.distances_[cluster2, k])
                elif self.linkage == 'complete':
                    new_distance = max(self.distances_[cluster1, k], self.distances_[cluster2, k])
                elif self.linkage == 'average':
                    size1 = len(self.clusters_[cluster1]) - len(self.clusters_[cluster2])
                    size2 = len(self.clusters_[cluster2])
                    new_distance = (self.distances_[cluster1, k] * size1 + self.distances_[cluster2, k] * size2) / (size1 + size2)
                
                self.distances_[cluster1, k] = self.distances_[k, cluster1] = new_distance

        # Remove the old cluster's distances
        self.distances_[:, cluster2] = self.distances_[cluster2, :] = np.inf

    def get_cluster_labels(self):
        """
        Assigns cluster labels to each point in the dataset.

        :return: numpy array of cluster labels
        """
        for idx, cluster in self.clusters_.items():
            for point in cluster:
                self.labels_[point] = idx
        return self.labels_
    

def plot_clusters(data, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('Agglomerative Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster label')
    plt.show()



X = LoadClassificationDataSet()

# single', 'complete', 'average
model = AgglomerativeClustering(data=X, n_clusters=15, linkage='complete')
model.fit()
labels = model.get_cluster_labels()

plot_clusters(X, labels)




