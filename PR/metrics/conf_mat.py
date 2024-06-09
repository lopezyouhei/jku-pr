import matplotlib.pyplot as plt
import os
import seaborn as sns

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_path = os.path.join(os.path.dirname(project_dir), "results")

def plot_confusion_matrix(conf_mat, 
                          class_names, 
                          title,
                          algorithm_name, 
                          category_name, 
                          model_name,
                          layer_name,
                          file_path=results_path):
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, 
                annot=True, 
                fmt='d',
                linewidths=.5,
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.savefig(os.path.join(file_path, 
                             f"{algorithm_name}_confusion_matrix_{category_name}_{model_name}_{layer_name}.png"))
    plt.close()