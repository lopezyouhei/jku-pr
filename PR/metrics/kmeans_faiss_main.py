import argparse
import csv
import faiss
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score
from sklearn.preprocessing import normalize
import sys
import torch

from metrics_utils import load_yaml, group_classes
from conf_mat import plot_confusion_matrix

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from config.metric_models import models # change imported model dictionary based on desired test
from config.class_group import *
from NNCLR.model import NNCLRHead

def run_faiss_clustering(data, n_clusters=1000):
    # Setting up the index - using the Flat index for cosine similarity
    d = data.shape[1]  # dimension
    index = faiss.IndexFlatIP(d)  # Inner Product (IP) is equivalent to cosine similarity when data is normalized

    # Setting up the KMeans clustering
    kmeans = faiss.Clustering(d, n_clusters)
    kmeans.verbose = True
    kmeans.niter = 25
    kmeans.nredo = 5 # would be better to increase this number but it takes too long since faiss-gpu cannot be used
    kmeans.min_points_per_centroid = 1
    kmeans.max_points_per_centroid = 100000

    # Train KMeans
    kmeans.train(data, index)

    # Assign the vectors to the clusters
    _, I = index.search(data, 1)  # Find the nearest cluster for each vector
    return I.flatten()  # Return the cluster labels

def hungarian_algorithm(gt_labels, predicted_labels):
    confusion = confusion_matrix(gt_labels, predicted_labels)
    rows, cols = linear_sum_assignment(-confusion)

    new_labels = np.zeros_like(predicted_labels)
    for row, col in zip(rows, cols):
        new_labels[predicted_labels == col] = row
    return new_labels

def main(device, categories):
    # get path
    data_path = os.path.join(os.path.dirname(project_dir), "features_mae_large")
    test_features_path = os.path.join(data_path, "mae_l23_cls_test_centercrop.th")
    test_labels_path = os.path.join(data_path, "test-labels.th")
    test_features = torch.load(test_features_path)
    test_labels = torch.load(test_labels_path)
    n_clusters = 1000

    # get synset categories
    if categories in ['main_5_categories', 'dog_15_categories', 'wen_10_categories']:
        if categories == 'main_5_categories':
            synset_categories = main_5_categories
            n_clusters = 5
        elif categories == 'dog_15_categories':
            synset_categories = dog_15_categories
            n_clusters = 15
        elif categories == 'wen_10_categories':
            synset_categories = wen_10_categories
            n_clusters = 10
        
        # load class_to_classid
        class_to_classid = load_yaml(os.path.join(data_path, "in1k_class_to_classid.yaml"))
        # get category names and labels
        category_names, category_labels = group_classes(test_labels, synset_categories, class_to_classid)
        # remove -1 labels from category_labels and test_features
        category_labels_tensor = torch.tensor(category_labels)
        valid_indices = category_labels_tensor != -1
        test_labels = category_labels_tensor[valid_indices].numpy()
        test_features = test_features[valid_indices]
    elif categories is None:
        test_labels = test_labels.numpy()
    else:
        raise ValueError(f"Invalid categories: {categories}")

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

        # apply FAISS clustering
        l_cluster_labels = run_faiss_clustering(normalized_l_embeddings, n_clusters)
        z_cluster_labels = run_faiss_clustering(normalized_z_embeddings, n_clusters)
        p_cluster_labels = run_faiss_clustering(normalized_p_embeddings, n_clusters)

        # align cluster labels using Hungarian algorithm
        l_aligned_labels = hungarian_algorithm(test_labels, l_cluster_labels)
        z_aligned_labels = hungarian_algorithm(test_labels, z_cluster_labels)
        p_aligned_labels = hungarian_algorithm(test_labels, p_cluster_labels)

        if categories is not None:
            # plot confusion matrix
            plot_confusion_matrix(confusion_matrix(test_labels, l_aligned_labels), 
                                category_names, 
                                f"Confusion Matrix for {model_name} using l embeddings",
                                "kmeans",
                                categories,
                                model_name,
                                "l",)
            plot_confusion_matrix(confusion_matrix(test_labels, z_aligned_labels), 
                                category_names, 
                                f"Confusion Matrix for {model_name} using z embeddings",
                                "kmeans",
                                categories,
                                model_name,
                                "z")
            plot_confusion_matrix(confusion_matrix(test_labels, p_aligned_labels), 
                                category_names, 
                                f"Confusion Matrix for {model_name} using p embeddings",
                                "kmeans",
                                categories,
                                model_name,
                                "p")

        # calculate accuracy
        l_accuracy = accuracy_score(test_labels, l_aligned_labels)
        z_accuracy = accuracy_score(test_labels, z_aligned_labels)
        p_accuracy = accuracy_score(test_labels, p_aligned_labels)

        # calculate silhouette scores directly on normalized embeddings using cosine similarity
        l_silhouette_score = silhouette_score(normalized_l_embeddings, l_cluster_labels, metric='cosine')
        z_silhouette_score = silhouette_score(normalized_z_embeddings, z_cluster_labels, metric='cosine')
        p_silhouette_score = silhouette_score(normalized_p_embeddings, p_cluster_labels, metric='cosine')


        # print(f"l_accuracy for {model_name}: {l_accuracy}")
        # print(f"z_accuracy for {model_name}: {z_accuracy}")
        # print(f"p_accuracy for {model_name}: {p_accuracy}\n")
        # print(f"l_silhouette_score for {model_name}: {l_silhouette_score}")
        # print(f"z_silhouette_score for {model_name}: {z_silhouette_score}")
        # print(f"p_silhouette_score for {model_name}: {p_silhouette_score}\n")

        # store results in a csv file
        csv_file = os.path.join(os.path.dirname(project_dir), 
                                "results", 
                                f"kmeans_{categories}_{model_name}.csv")
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["model", "l_accuracy", "z_accuracy", "p_accuracy", "l_silhouette_score", "z_silhouette_score", "p_silhouette_score"])
            writer.writerow([model_name, l_accuracy, z_accuracy, p_accuracy, l_silhouette_score, z_silhouette_score, p_silhouette_score])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--categories', type=str, default=None, help='Categories from class_group.py')
    args = parser.parse_args()
    main(args.device, args.categories)
