import argparse
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import sys
import torch

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from config.metric_models import models # change imported model dictionary based on desired test
from NNCLR.model import NNCLRHead

def main(device):

    # get path
    # TODO: functionalize this
    data_path = os.path.join(os.path.dirname(project_dir), "features_mae_large")
    test_features_path = os.path.join(data_path, "mae_l23_cls_test_centercrop.th")
    test_labels_path = os.path.join(data_path, "test-labels.th")
    test_features = torch.load(test_features_path)
    test_labels = torch.load(test_labels_path).numpy()

    # TODO: functionalize this
    for model_name, model_path in models.items():
        # load model
        model = NNCLRHead()
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        # set model to evaluation mode
        model.eval()
        # get embeddings
        with torch.inference_mode():
            l_embeddings = []
            z_embeddings = []
            p_embeddings = []
            for x in test_features:
                x = x.unsqueeze(0).to(device)
                l, z, p = model(x)
                l_embeddings.append(l.cpu())
                z_embeddings.append(z.cpu())
                p_embeddings.append(p.cpu())
            l_embeddings = torch.cat(l_embeddings, axis=0)
            z_embeddings = torch.cat(z_embeddings, axis=0)
            p_embeddings = torch.cat(p_embeddings, axis=0)

        # normalize embeddings
        normalized_l_embeddings = normalize(l_embeddings.numpy(), norm='l2', axis=1)
        normalized_z_embeddings = normalize(z_embeddings.numpy(), norm='l2', axis=1)
        normalized_p_embeddings = normalize(p_embeddings.numpy(), norm='l2', axis=1)

        # calculate cosine similarity
        l_similarity = cosine_similarity(normalized_l_embeddings)
        z_similarity = cosine_similarity(normalized_z_embeddings)
        p_similarity = cosine_similarity(normalized_p_embeddings)

        # apply KMeans clustering on cosine similarity
        # kmeans = KMeans(n_clusters=100, random_state=42) # leads to memory error
        kmeans = MiniBatchKMeans(n_clusters=100, random_state=42, batch_size=1024)
        l_cluster_labels = kmeans.fit_predict(l_similarity)
        z_cluster_labels = kmeans.fit_predict(z_similarity)
        p_cluster_labels = kmeans.fit_predict(p_similarity)

        # calculate confusion matrix
        l_confusion_matrix = confusion_matrix(test_labels, l_cluster_labels)
        z_confusion_matrix = confusion_matrix(test_labels, z_cluster_labels)
        p_confusion_matrix = confusion_matrix(test_labels, p_cluster_labels)

        # Hungarian algorithm
        # TODO: functionalize this
        rows, cols = linear_sum_assignment(-l_confusion_matrix)
        new_l_cluster_labels = np.zeros_like(l_cluster_labels)
        for row, col in zip(rows, cols):
            new_l_cluster_labels[l_cluster_labels == col] = row

        rows, cols = linear_sum_assignment(-z_confusion_matrix)
        new_z_cluster_labels = np.zeros_like(z_cluster_labels)
        for row, col in zip(rows, cols):
            new_z_cluster_labels[z_cluster_labels == col] = row

        rows, cols = linear_sum_assignment(-p_confusion_matrix)
        new_p_cluster_labels = np.zeros_like(p_cluster_labels)
        for row, col in zip(rows, cols):
            new_p_cluster_labels[p_cluster_labels == col] = row

        # calculate silhouette score
        l_silhouette_score = silhouette_score(l_similarity, new_l_cluster_labels, metric='cosine')
        z_silhouette_score = silhouette_score(z_similarity, new_z_cluster_labels, metric='cosine')
        p_silhouette_score = silhouette_score(p_similarity, new_p_cluster_labels, metric='cosine')

        print(f"l_silhouette_score for {model_name}: {l_silhouette_score}")
        print(f"z_silhouette_score for {model_name}: {z_silhouette_score}")
        print(f"p_silhouette_score for {model_name}: {p_silhouette_score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()
    main(args.device)
