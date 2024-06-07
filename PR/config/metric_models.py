import os

model_path = os.path.join(os.getcwd(), "models")

models = {
    'baseline': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_9_1-0_baseline.pth'),
    'mae-ct': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_9_1-0_mae-ct_baseline.pth'),
    'reduced-100': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_10_0-9_reduced_classes.pth'),
    'reduced-200': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_11_0-8_reduced_classes.pth'),
    'reduced-300': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_13_0-7_reduced_classes.pth'),
    'reduced-400': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_16_0-6_reduced_classes.pth'),
    'reduced-500': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_19_0-5_reduced_classes.pth'),
    'reduced-600': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_24_0-4_reduced_classes.pth'),
    'reduced-700': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_32_0-3_reduced_classes.pth'),
    'reduced-800': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_49_0-2_reduced_classes.pth'),
    'reduced-900': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_99_0-1_reduced_classes.pth'),
}

best_models = {
    'baseline': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_9_1-0_baseline.pth'),
    'mae-ct': os.path.join(model_path, 'tensors_v1_seed_0_tensors_v2_seed_0_9_1-0_mae-ct_baseline.pth'),
    'reduced-xxx':"",
}