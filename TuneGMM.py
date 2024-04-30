import numpy as np
from GMM import GaussianMixtureModel
from GMM import plot_gmm
from LoadClassificationDataSet import LoadClassificationDataSet
import matplotlib.pyplot as plt

X = LoadClassificationDataSet()

best_gmm = None
best_log_likelihood = -np.inf  # Start with the worst log likelihood

n_components = 3

for i in range(20):  # Number of initializations
    gmm = GaussianMixtureModel(n_components=n_components, max_iter=100)
    gmm.fit(X)
    if gmm.log_likelihood > best_log_likelihood:
        print(i)
        best_log_likelihood = gmm.log_likelihood
        best_gmm = gmm

# Create a new GMM instance to use with the best parameters
final_gmm = GaussianMixtureModel(n_components=n_components)
final_gmm.set_parameters(best_gmm.means, best_gmm.covariances, best_gmm.weights, best_gmm.log_likelihood)

# Now you can use final_gmm for further analysis or visualization
labels = final_gmm.predict(X)
responsibilities = final_gmm._e_step(X)

# Assuming the plot_gmm function has been appropriately defined to use these parameters
plt.figure(figsize=(10, 8))
ax = plt.gca()
plot_gmm(X, final_gmm.means, final_gmm.covariances, responsibilities, ax=ax)
plt.title('Gaussian Mixture Model Clustering with Best Parameters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


