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

device = "cuda" if torch.cuda.is_available() else "cpu"

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
train_indices = torch.randperm(len(train_labels))[:int(0.1*len(train_labels))]
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
        train_embeddings = []
        for x in train_features:
            x = x.unsqueeze(0).to(device)
            _, _, p = model(x)
            train_embeddings.append(p.cpu())
        train_embeddings = torch.cat(train_embeddings, axis=0)
        train_emb = train_embeddings.numpy()
        train_x = normalize(train_emb, norm='l2', axis=1)

        test_embeddings = []
        for x in test_features:
            x = x.unsqueeze(0).to(device)
            _, _, p = model(x)
            test_embeddings.append(p.cpu())
        test_embeddings = torch.cat(test_embeddings, axis=0)
        test_emb = test_embeddings.numpy()
        test_x = normalize(test_emb, norm='l2', axis=1)
    
    # fit k-NN classifier
    N_NEIGHBORS = 10
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='cosine')
    knn.fit(train_x, train_labels.numpy())
    # predict labels
    pred = knn.predict(test_x)
    # calculate accuracy
    acc = (pred == test_labels.numpy()).mean()
    # store results
    results[model_name] = acc

# store results in a csv file with used N_NEIGHBORS
csv_file = os.path.join(os.path.dirname(project_dir), 
                        "results", 
                        f"knn_{N_NEIGHBORS}.csv")
with open(csv_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["model", "accuracy"])
    for model_name, acc in results.items():
        writer.writerow([model_name, acc])