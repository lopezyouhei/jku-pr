import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import List

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_path = os.path.join(os.path.dirname(project_dir), "results")

def plot_umap(embedding : np.ndarray, 
              category_labels : torch.Tensor,
              category_names : List, 
              synset_category_name : str, 
              model_name : str, 
              layer_name : str):
    fig, ax = plt.subplots()
    unique_labels = np.unique(category_labels)

    # colors = plt.cm.get_cmap('gist_ncar', len(unique_labels))
    colors = plt.cm.get_cmap('tab20', 20)

    for i, label in enumerate(unique_labels):
        ax.scatter(embedding[:, 0][category_labels == label], 
                   embedding[:, 1][category_labels == label], 
                   color=colors(i), 
                   label=f'{category_names[int(label)].capitalize()}', 
                   s = 5)

    # ax.legend(title="Labels")
    ax.legend(title="Labels", loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              fancybox=True, shadow=True, ncol=len(unique_labels)//2)
    filename = os.path.join(results_path, f"{model_name}_layer-{layer_name}_{synset_category_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)