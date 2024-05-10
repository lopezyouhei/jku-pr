import os
import random
from itertools import combinations
from PR.nnclr_train import train

data_path = os.path.join(os.getcwd(), "features_mae_large")

views_list = []
for file in os.listdir(data_path):
    if "v1_seed" in file or "v2_seed" in file:
        views_list.append(os.path.join(data_path, file).replace("\\", "/"))

def generate_random_pairs(num_pairs=15):
    all_pairs = list(combinations(views_list, 2))
    random.shuffle(all_pairs)
    return all_pairs[:num_pairs]

def train_random_pairs(num_pairs=15):
    pairs = generate_random_pairs(num_pairs)
    for i, pair in enumerate(pairs):
        print(f"Training pair {i+1}/{num_pairs}: {pair}")
        train(pair[0], pair[1])

