from flask import Flask, request, jsonify
import pandas as pd
from KNN import KMeans
from GMM import GaussianMixtureModel
from LoadClassificationDataSet import LoadClassificationDataSetChosen

app = Flask(__name__)

def load_dataset(dataset_name):
    return pd.read_csv(f'datasets/{dataset_name}')

@app.route('/knn', methods=['POST'])
def knn_clustering():
    if request.is_json:
        data = request.get_json()
        X = LoadClassificationDataSetChosen(data['dataset'])
        kmeans = KMeans(k=data['k'])
        kmeans.fit(X)
        labels = kmeans.labels
        return jsonify({"status": "success", "labels": labels.tolist()})
    else:
        return jsonify({"error": "Request must be JSON"}), 400

@app.route('/gmm', methods=['POST'])
def gmm_clustering():
    if request.is_json:
        data = request.get_json()
        X = LoadClassificationDataSetChosen(data['dataset'])
        gmm = GaussianMixtureModel(n_components=data['n_components'], max_iter=data['max_iter'])
        gmm.fit(X)
        responsibilities = gmm._e_step(X)
        means = gmm.means
        covariances = gmm.covariances
        return jsonify({
            "status": "success",
            "responsibilities": responsibilities.tolist(),
            "means": means.tolist(),
            "covariances": covariances.tolist()
        })
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)


