import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score


def cluster_and_save(data_path, output_path, n_clusters, standardize=True):
    """
    Load data from CSV, perform optional standardization, cluster with KMeans,
    and save cluster labels to output_path in 'id,label' format.
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.values

    # Standardize features
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Optional PCA for dimensionality reduction
    # pca = PCA(n_components=min(X.shape[1], 10))
    # X = pca.fit_transform(X)

    # Fit KMeans with explicit hyperparameters documented
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42
    )
    labels = kmeans.fit_predict(X)

    # Prepare submission DataFrame with id preserving original order
    out_df = pd.DataFrame({
        'id': np.arange(len(labels)),
        'label': labels
    })
    out_df.to_csv(output_path, index=False)
    print(f"Saved clustering to {output_path} (n_clusters={n_clusters})")


if __name__ == "__main__":
    # Public dataset clustering
    public_path = 'public_data.csv'
    pub_dim = pd.read_csv(public_path).shape[1]
    n_pub = 4 * pub_dim - 1
    cluster_and_save(public_path, 'public_submission.csv', n_pub)

    # Private dataset clustering
    private_path = 'private_data.csv'
    priv_dim = pd.read_csv(private_path).shape[1]
    n_priv = 4 * priv_dim - 1
    cluster_and_save(private_path, 'private_submission.csv', n_priv)

    # Example evaluation (if ground truth available)
    try:
        truth = pd.read_csv('public_ground_truth.csv')
        fmi = fowlkes_mallows_score(truth.values.ravel(), pd.read_csv('public_submission.csv')['label'])
        print(f"Public dataset FMI: {fmi:.4f}")
    except FileNotFoundError:
        print("Ground truth file for public dataset not found. Skipping evaluation.")
