import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from LoadClassificationDataSet import LoadClassificationDataSet


class GaussianMixtureModel:
    def __init__(self, n_components=2, tol=1e-6, max_iter=100):
        self.n_components = n_components  # Number of clusters
        self.tol = tol  # Tolerance to declare convergence
        self.max_iter = max_iter  # Maximum number of iterations
        self.means = None  # Means of the Gaussians
        self.covariances = None  # Covariances of the Gaussians
        self.weights = None  # Mixing weights
        self.log_likelihood = -np.inf  # Log likelihood of the model

    def set_parameters(self, means, covariances, weights, log_likelihood):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.log_likelihood = log_likelihood

    def fit(self, X):
        # Initialize parameters
        self._initialize_parameters(X)

        initial_log_likelihood = self._compute_log_likelihood(X)
        # print(f"Initial log likelihood: {initial_log_likelihood}")

        for i in range(self.max_iter):
            print(f"Iteration {i+1}")

            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Compute log likelihood
            current_log_likelihood = self._compute_log_likelihood(X)
            print(f"Log Likelihood: {current_log_likelihood}")
            print(f"Change in Log Likelihood: {abs(current_log_likelihood - self.log_likelihood)}")

            # Check for convergence
            if abs(current_log_likelihood - self.log_likelihood) < self.tol:
                print("Convergence reached.")
                self.last_responsibilities = self._e_step(X)
                break

            self.log_likelihood = current_log_likelihood
            self.last_responsibilities = self._e_step(X)
        return self


    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)


    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape

        # Initialize means using K-means++ algorithm
        self.means = np.empty((self.n_components, n_features))
        initial_idx = np.random.choice(n_samples)
        self.means[0] = X[initial_idx]

        for k in range(1, self.n_components):
            dist_sq = np.array([min([np.inner(x-m, x-m) for m in self.means[:k]]) for x in X])
            probabilities = dist_sq / dist_sq.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    self.means[k] = X[i]
                    break

        # Initialize covariances to identity matrices
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])

        # Initialize weights uniformly
        self.weights = np.full(self.n_components, 1 / self.n_components)


    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            # Calculate the multivariate normal density for component k
            cov_inv = np.linalg.inv(self.covariances[k])
            diff = X - self.means[k]
            exp_component = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
            norm_const = 1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(self.covariances[k]))
            density = norm_const * exp_component

            # Multiply by the weight of the component
            responsibilities[:, k] = self.weights[k] * density

        # Normalize responsibilities so that they sum to 1 for each sample
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= responsibilities_sum

        return responsibilities


    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape

        # Update means
        self.means = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            weights_sum = responsibilities[:, k].sum()
            self.means[k] = (X * responsibilities[:, k, np.newaxis]).sum(axis=0) / weights_sum

        # Update covariances
        self.covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_diff = diff.T * responsibilities[:, k]
            self.covariances[k] = np.dot(weighted_diff, diff) / responsibilities[:, k].sum()

        # Update mixing weights
        self.weights = responsibilities.sum(axis=0) / n_samples


    # def _compute_log_likelihood(self, X):
    #     log_likelihood = 0
    #     for k in range(self.n_components):
    #         # Calculate the multivariate normal density for component k
    #         cov_inv = np.linalg.inv(self.covariances[k])
    #         diff = X - self.means[k]
    #         exp_component = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
    #         norm_const = 1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(self.covariances[k]))
    #         density = norm_const * exp_component

    #         # Add the log of the sum of weighted densities to the log likelihood
    #         log_likelihood += np.log(self.weights[k] * density + 1e-10)  # Add a small constant to prevent log(0)

    #     self.log_likelihood = np.sum(log_likelihood)
    #     return self.log_likelihood
    

    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        n_samples = X.shape[0]
        total_log_likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            # Calculate the inverse and determinant of covariance matrix
            cov_inv = np.linalg.inv(self.covariances[k])
            cov_det = np.linalg.det(self.covariances[k])

            # Calculate the norm constant for the Gaussian distribution
            norm_const = 1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * cov_det)

            # Calculate the difference from the mean
            diff = X - self.means[k]

            # Calculate the exponential component of the Gaussian formula
            exp_component = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))

            # Calculate the density
            density = norm_const * exp_component

            # Accumulate the weighted densities for this component
            total_log_likelihood += self.weights[k] * density

        # Take the log of the sum of weighted densities to compute the final log likelihood
        log_likelihood = np.sum(np.log(total_log_likelihood + 1e-10))  # Small constant to prevent log(0)

        self.log_likelihood = log_likelihood
        return self.log_likelihood



def plot_gmm(X, means, covariances, responsibilities, ax=None):
    if ax is None:
        ax = plt.gca()
    
    # Create a scatter plot where each point is colored by its highest responsibility
    # Color intensity will indicate the certainty of belonging to the cluster
    cluster_membership = np.argmax(responsibilities, axis=1)
    colors = responsibilities.max(axis=1)  # Take the maximum responsibility as the color intensity
    scatter = ax.scatter(X[:, 0], X[:, 1], c=cluster_membership, cmap='viridis', s=30, alpha=0.5, edgecolor='k', linewidth=0.5, zorder=2)
    ax.axis('equal')
    
    # Color bar to show the probability scale
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster Membership Probability')

    # Plotting ellipses to represent Gaussian components
    for pos, covar in zip(means, covariances):
        eigvals, eigvecs = np.linalg.eigh(covar)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
        theta = np.arctan2(vy, vx)
        for std_dev in range(1, 3):  # Draw 1-std-dev and 2-std-dev ellipses
            ell = Ellipse(pos, 2 * np.sqrt(eigvals[0]) * std_dev, 2 * np.sqrt(eigvals[1]) * std_dev,
                          angle=np.degrees(theta), alpha=0.5/std_dev, edgecolor='red', facecolor='none')
            ax.add_artist(ell)


# X = LoadClassificationDataSet()

# # Create a GMM instance
# gmm = GaussianMixtureModel(n_components=15, max_iter=300)
# gmm.fit(X)

# responsibilities = gmm._e_step(X)

# means = gmm.means
# covariances = gmm.covariances

# plt.figure(figsize=(10, 8))
# ax = plt.gca()
# plot_gmm(X, means, covariances, responsibilities, ax=ax)
# plt.title('Gaussian Mixture Model Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()














