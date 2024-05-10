import random
import os
import torch
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def generate_pair(random_bool, data_path, model_path):
    views_list = []
    for file in os.listdir(data_path):
        if "v1_seed" in file or "v2_seed" in file:
            views_list.append(os.path.join(data_path, file).replace("\\", "/"))
    # generate random pair which is not in model_path
    if random_bool:
        view1, view2 = random.sample(views_list, 2)
        view1_name = view1.split("/")[-1].split(".")[0]
        view2_name = view2.split("/")[-1].split(".")[0]
        if f"best_model_{view1_name}_{view2_name}.pth" in os.listdir(model_path):
            return generate_pair(random_bool, data_path, model_path)
        return view1, view2
    elif not random_bool:
        view1 = os.path.join(data_path, "tensors_v1_seed_0.th").replace("\\", "/")
        view2 = os.path.join(data_path, "tensors_v2_seed_0.th").replace("\\", "/")
        return view1, view2