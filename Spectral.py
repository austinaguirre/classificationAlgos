import numpy as np
from LoadClassificationDataSet import LoadClassificationDataSet
from GMM import GaussianMixtureModel as GMM
from KNN import KMeans
import matplotlib.pyplot as plt


class SpectralClustering:
    def __init__(self, n_clusters=2, sigma=1.0):
        self.n_clusters = n_clusters  # number of clusters
        self.sigma = sigma

    def fit(self, X):
        # Main method to fit the model
        self._compute_similarity_matrix(X)
        self._compute_laplacian_matrix()
        self._compute_eigen_decomposition()
        self._cluster_data()

    def _compute_similarity_matrix(self, X):
        # Compute the Gaussian similarity matrix based on input data X
        n_samples = X.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                squared_distance = np.sum((X[i] - X[j]) ** 2)
                similarity_matrix[i, j] = np.exp(-squared_distance / (2 * (self.sigma ** 2)))
        self.similarity_matrix = similarity_matrix

    def _compute_laplacian_matrix(self):
        # Compute the Laplacian matrix from the similarity matrix
        n_samples = self.similarity_matrix.shape[0]
        degree_matrix = np.diag(np.sum(self.similarity_matrix, axis=1))
        self.laplacian_matrix = degree_matrix - self.similarity_matrix

    def _compute_eigen_decomposition(self):
        # Compute eigenvalues and eigenvectors for the Laplacian matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian_matrix)
        # Order the eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Store only the eigenvectors corresponding to the smallest eigenvalues
        self.eigenvectors = eigenvectors[:, :self.n_clusters]

    def _cluster_data(self):
        kmeans = KMeans(k=self.n_clusters)
        kmeans.fit(self.eigenvectors)
        self.labels_ = kmeans.labels

    def plot_clusters(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, cmap='viridis', marker='o')
        plt.title('Spectral Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster Label')
        plt.show()


X = LoadClassificationDataSet()

spectral = SpectralClustering(n_clusters=15, sigma=1.0)
spectral.fit(X)
spectral.plot_clusters(X)


