# Import necessary libraries
from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_gaussian_quantiles

def makedf(X, y, name):
    # Convert to DataFrame for visualization
    df = pd.DataFrame(X, columns=['X', 'Y'])
    df['Cluster'] = y  # You might not always have this if you're truly clustering

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='X', y='Y', hue='Cluster', palette='viridis', style='Cluster', s=100)
    plt.title('Scatter Plot of ' + name)
    plt.show()

    df.to_csv(name + '_classification_dataset.csv', index=False)


def makeBlobs ():
    X, y = make_blobs(n_samples=600, centers=15, n_features=2, random_state=42, cluster_std=1.2, center_box=(-20,20))
    makedf(X, y, "blobs")

def makeMoons():
    X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    makedf(X, y, "moons")

def makeMultiLabel():
    X, y = make_multilabel_classification(n_samples=100, n_features=2, n_classes=3, n_labels=2, random_state=42)
    makedf(X, y, "multilabel")

def makeGassianQuantiles():
    X, y = make_gaussian_quantiles(mean=None, cov=1.0, n_samples=500, n_features=2, n_classes=5, random_state=42)
    makedf(X, y, "gassianquantiles")

def makeCircles():
    X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
    makedf(X, y, "circles")


makeCircles()











