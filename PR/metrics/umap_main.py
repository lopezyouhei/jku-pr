import argparse
import os
import sys
import torch
import umap
import umap.plot

from metrics_utils import load_yaml, group_classes
from umap_plot import plot_umap

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from config.metric_models import models
from config.class_group import *
from NNCLR.model import NNCLRHead

def main(device, categories):

    # get synset categories
    if categories == 'main_5_categories':
        synset_categories = main_5_categories
    elif categories == 'dog_15_categories':
        synset_categories = dog_15_categories
    elif categories == 'wen_10_categories':
        synset_categories = wen_10_categories
    else:
        raise ValueError(f"Invalid categories: {categories}")

    # get path
    data_path = os.path.join(os.path.dirname(project_dir), "features_mae_large")
    test_features_path = os.path.join(data_path, "mae_l23_cls_test_centercrop.th")
    test_labels_path = os.path.join(data_path, "test-labels.th")
    test_features = torch.load(test_features_path)
    test_labels = torch.load(test_labels_path)

    # load class_to_classid
    class_to_classid = load_yaml(os.path.join(data_path, "in1k_class_to_classid.yaml"))
    # get category names and labels
    category_names, category_labels = group_classes(test_labels, synset_categories, class_to_classid)
    # remove -1 labels from category_labels and test_features
    category_labels_tensor = torch.tensor(category_labels)
    valid_indices = category_labels_tensor != -1
    category_labels = category_labels_tensor[valid_indices]
    test_features = test_features[valid_indices]

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
        
        # umap embedding
        reducer = umap.UMAP(random_state=42, n_jobs=1) # random state fixed for reproducibility
        l_umap = reducer.fit_transform(l_embeddings) # np.ndarray
        z_umap = reducer.fit_transform(z_embeddings) # np.ndarray
        p_umap = reducer.fit_transform(p_embeddings) # np.ndarray
        
        # plot umap
        plot_umap(l_umap, category_labels, category_names, categories, model_name, 'l')
        plot_umap(z_umap, category_labels, category_names, categories, model_name, 'z')
        plot_umap(p_umap, category_labels, category_names, categories, model_name, 'p')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UMAP Embedding")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--categories', type=str, default='main_5_categories', help='Categories from class_group.py')
    args = parser.parse_args()
    main(args.device, args.categories)