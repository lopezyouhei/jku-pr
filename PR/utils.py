import random
import os
import torch
    
def generate_pair(random_bool, data_path, model_path):
    views_list = []
    for file in os.listdir(data_path):
        if "v1_seed" in file or "v2_seed" in file:
            views_list.append(os.path.join(data_path, file).replace("\\", "/"))
    # generate random pair which is not in model_path
    if random_bool:
        # for loop to avoid infinite loop
        for _ in range(100):
            view1, view2 = random.sample(views_list, 2)
            view1_name = view1.split("/")[-1].split(".")[0]
            view2_name = view2.split("/")[-1].split(".")[0]
            model_name_prefix = f"{view1_name}_{view2_name}_"
            # check if model already exists
            model_exists = any(
                filename.startswith(model_name_prefix) and filename.endswith(".pth")
                for filename in os.listdir(model_path)
            )
            if not model_exists:
                return view1, view2
        raise Exception("No new pair found in 100 iterations, try again or increase range.")
    else:
        view1 = os.path.join(data_path, "tensors_v1_seed_0.th").replace("\\", "/")
        view2 = os.path.join(data_path, "tensors_v2_seed_0.th").replace("\\", "/")
        return view1, view2