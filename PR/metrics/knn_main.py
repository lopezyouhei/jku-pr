import argparse
import csv
import os
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
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

def main(n_neighbors, device, categories):
    # get paths
    data_path = os.path.join(os.path.dirname(project_dir), "features_mae_large")
    train_features_path = os.path.join(data_path, "mae_l23_cls_train_centercrop.th")
    train_labels_path = os.path.join(data_path, "train-labels.th")
    test_features_path = os.path.join(data_path, "mae_l23_cls_test_centercrop.th")
    test_labels_path = os.path.join(data_path, "test-labels.th")

    # load data
    # randomly sample 10% of training data and use as training set for k-NN classifier
    # sampling needs to be deterministic
    torch.manual_seed(42)
    train_features = torch.load(train_features_path)
    train_labels = torch.load(train_labels_path)
    train_indices = torch.randperm(len(train_labels))[:int(0.1 * len(train_labels))]
    train_features = train_features[train_indices]
    train_labels = train_labels[train_indices]
    test_features = torch.load(test_features_path)
    test_labels = torch.load(test_labels_path)

    # get synset categories
    if categories in ['main_5_categories', 'dog_15_categories', 'wen_10_categories']:
        if categories == 'main_5_categories':
            synset_categories = main_5_categories
        elif categories == 'dog_15_categories':
            synset_categories = dog_15_categories
        elif categories == 'wen_10_categories':
            synset_categories = wen_10_categories
        
        # load class_to_classid
        class_to_classid = load_yaml(os.path.join(data_path, "in1k_class_to_classid.yaml"))
        # get category names and labels
        category_names, category_labels = group_classes(test_labels, synset_categories, class_to_classid)
        # remove -1 labels from category_labels and test_features
        category_labels_tensor = torch.tensor(category_labels)
        valid_indices = category_labels_tensor != -1
        test_labels = category_labels_tensor[valid_indices]
        test_features = test_features[valid_indices]
    elif categories is None:
        pass
    else:
        raise ValueError(f"Invalid categories: {categories}")

    results = {}

    for model_name, model_path in models.items():
        # load model
        model = NNCLRHead()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        # set model to evaluation mode
        model.eval()
        with torch.inference_mode():
            l_embeddings = []
            z_embeddings = []
            p_embeddings = []
            for x in train_features:
                x = x.unsqueeze(0).to(device)
                l, z, p = model(x)
                l_embeddings.append(l.cpu())
                z_embeddings.append(z.cpu())
                p_embeddings.append(p.cpu())
            l_train_embeddings = torch.cat(l_embeddings, axis=0)
            z_train_embeddings = torch.cat(z_embeddings, axis=0)
            p_train_embeddings = torch.cat(p_embeddings, axis=0)
            normalized_l_train_embeddings = normalize(l_train_embeddings.numpy(), norm='l2', axis=1)
            normalized_z_train_embeddings = normalize(z_train_embeddings.numpy(), norm='l2', axis=1)
            normalized_p_train_embeddings = normalize(p_train_embeddings.numpy(), norm='l2', axis=1)

            for x in test_features:
                x = x.unsqueeze(0).to(device)
                l, z, p = model(x)
                l_embeddings.append(l.cpu())
                z_embeddings.append(z.cpu())
                p_embeddings.append(p.cpu())
            l_test_embeddings = torch.cat(l_embeddings, axis=0)
            z_test_embeddings = torch.cat(z_embeddings, axis=0)
            p_test_embeddings = torch.cat(p_embeddings, axis=0)
            normalized_l_embeddings = normalize(l_test_embeddings.numpy(), norm='l2', axis=1)
            normalized_z_embeddings = normalize(z_test_embeddings.numpy(), norm='l2', axis=1)
            normalized_p_embeddings = normalize(p_test_embeddings.numpy(), norm='l2', axis=1)

            train_embeddings = [model(x.unsqueeze(0).to(device))[2].cpu() for x in train_features]
            train_embeddings = torch.cat(train_embeddings, axis=0)
            train_x = normalize(train_embeddings.numpy(), norm='l2', axis=1)

            test_embeddings = [model(x.unsqueeze(0).to(device))[2].cpu() for x in test_features]
            test_embeddings = torch.cat(test_embeddings, axis=0)
            test_x = normalize(test_embeddings.numpy(), norm='l2', axis=1)

        # fit k-NN classifier
        l_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        l_knn.fit(normalized_l_train_embeddings, train_labels.numpy())
        z_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        z_knn.fit(normalized_z_train_embeddings, train_labels.numpy())
        p_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        p_knn.fit(normalized_p_train_embeddings, train_labels.numpy())
        # predict labels
        l_pred = l_knn.predict(normalized_l_embeddings)
        z_pred = z_knn.predict(normalized_z_embeddings)
        p_pred = p_knn.predict(normalized_p_embeddings)
        # calculate accuracy
        l_acc = (l_pred == test_labels.numpy()).mean()
        z_acc = (z_pred == test_labels.numpy()).mean()
        p_acc = (p_pred == test_labels.numpy()).mean()
        # store results
        results[model_name] = [l_acc, z_acc, p_acc]

        if categories is not None:
            # plot confusion matrix
            plot_confusion_matrix(confusion_matrix(test_labels, l_pred), 
                                category_names, 
                                f"Confusion Matrix for {model_name} using l embeddings",
                                "knn",
                                categories,
                                model_name,
                                "l",)
            plot_confusion_matrix(confusion_matrix(test_labels, z_pred),
                                category_names, 
                                f"Confusion Matrix for {model_name} using z embeddings",
                                "knn",
                                categories,
                                model_name,
                                "z")
            plot_confusion_matrix(confusion_matrix(test_labels, p_pred),
                                category_names, 
                                f"Confusion Matrix for {model_name} using p embeddings",
                                "knn",
                                categories,
                                model_name,
                                "p")
            

    # store results in a csv file with used n_neighbors
    csv_file = os.path.join(os.path.dirname(project_dir), 
                        "results", 
                        f"knn_{n_neighbors}_{categories}.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "l_accuracy", "z_accuracy", "p_accuracy"])
        for model_name, acc in results.items():
            writer.writerow([model_name, acc[0], acc[1], acc[2]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="k-NN Classification")
    parser.add_argument("--n_neighbors", type=int, default=13, help="Number of neighbors for k-NN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--categories", type=str, default=None, help="Categories to group classes")
    args = parser.parse_args()

    main(args.n_neighbors, args.device, args.categories)