import argparse
import csv
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import sys
import torch

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from config.metric_models import models # change imported model dictionary based on desired test
from NNCLR.model import NNCLRHead

def main(n_neighbors, device):
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

    results = {}

    for model_name, model_path in models.items():
        # load model
        model = NNCLRHead()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        # set model to evaluation mode
        model.eval()
        with torch.inference_mode():
            train_embeddings = [model(x.unsqueeze(0).to(device))[2].cpu() for x in train_features]
            train_embeddings = torch.cat(train_embeddings, axis=0)
            train_x = normalize(train_embeddings.numpy(), norm='l2', axis=1)

            test_embeddings = [model(x.unsqueeze(0).to(device))[2].cpu() for x in test_features]
            test_embeddings = torch.cat(test_embeddings, axis=0)
            test_x = normalize(test_embeddings.numpy(), norm='l2', axis=1)

        # fit k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        knn.fit(train_x, train_labels.numpy())
        # predict labels
        pred = knn.predict(test_x)
        # calculate accuracy
        acc = (pred == test_labels.numpy()).mean()
        # store results
        results[model_name] = acc

    # store results in a csv file with used n_neighbors
    csv_file = os.path.join(os.path.dirname(project_dir), 
                        "results", 
                        f"knn_{n_neighbors}.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "accuracy"])
        for model_name, acc in results.items():
            writer.writerow([model_name, acc])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="k-NN Classification")
    parser.add_argument("--n_neighbors", type=int, default=9, help="Number of neighbors for k-NN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    main(args.n_neighbors, args.device)