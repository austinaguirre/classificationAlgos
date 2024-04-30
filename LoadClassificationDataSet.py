import numpy as np


def LoadClassificationDataSet():
    file_path = 'blobs_classification_dataset.csv'

    # Load the data while skipping the first row (header) and forcing the data into float type for the x and y, and int for Cluster
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=[('x', float), ('y', float), ('Cluster', int)])

    # Access the x, y, and Cluster columns
    x = data['x']
    y = data['y']
    clusters = data['Cluster']

    # If you need to combine x and y into a single 2D array for further processing (e.g., for GMM input), you can do:
    X = np.column_stack((x, y))

    return X

def LoadClassificationDataSetChosen(filePath):
    file_path = filePath

    # Load the data while skipping the first row (header) and forcing the data into float type for the x and y, and int for Cluster
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=[('x', float), ('y', float), ('Cluster', int)])

    # Access the x, y, and Cluster columns
    x = data['x']
    y = data['y']
    clusters = data['Cluster']

    # If you need to combine x and y into a single 2D array for further processing (e.g., for GMM input), you can do:
    X = np.column_stack((x, y))

    return X


# X = LoadClassificationDataSet()
# print(X)


