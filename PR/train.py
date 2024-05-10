from lightly.loss import NTXentLoss
from lightly.models.modules import NNMemoryBankModule
import math
import os
import torch
from torch.utils.data import DataLoader
import wandb

from config.model_config import *
from utils import generate_pair
from NNCLR import MAEDataset, NNCLRHead

WORKING_DIR = os.getcwd()
DATA_PATH = os.path.join(WORKING_DIR, "features_mae_large")
MODEL_PATH = os.path.join(WORKING_DIR, "models")

labels_path = os.path.join(DATA_PATH, "train-labels.th")

def train(tags, random_views=True, reduce_factor=1.0):
    with wandb.init(project="NNCLR", tags=tags) as run:
        config = wandb.config
        config.epochs = config.get("epochs", EPOCHS)
        config.batch_size = config.get("batch_size", BATCH_SIZE)
        config.lr = config.get("lr", LR)
        config.temperature = config.get("temperature", TEMPERATURE)
        config.queue_size = config.get("queue_size", QUEUE_SIZE)
        config.weight_decay = config.get("weight_decay", WEIGHT_DECAY)
        config.project_hidden_dim = config.get("project_hidden_dim", PROJECT_HIDDEN_DIM)
        config.project_output_dim = config.get("project_output_dim", PROJECT_OUTPUT_DIM)
        config.prediction_hidden_dim = config.get("prediction_hidden_dim", PREDICTION_HIDDEN_DIM)
        config.prediction_output_dim = config.get("prediction_output_dim", PREDICTION_OUTPUT_DIM)
        config.random_views = config.get("random_views", random_views)
        config.reduce_factor = config.get("reduce_factor", reduce_factor)
        config.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # create a dataset
        # TODO: write function which takes boolean to generate random pairs or return a given pair
        dataset = None
        view1_path, view2_path = generate_pair(config.random_views, 
                                               DATA_PATH, 
                                               MODEL_PATH)
        
        view1_name = view1_path.split("/")[-1].split(".")[0]
        view2_name = view2_path.split("/")[-1].split(".")[0]

        dataset = MAEDataset(view1_path, 
                             view2_path,
                             labels_path,
                             config.reduce_factor)

        # create the dataloader
        dataloader = DataLoader(dataset, 
                                batch_size=config.batch_size, 
                                shuffle=True)

        # create the NNCLR model
        model = NNCLRHead(config.project_hidden_dim, 
                          config.project_output_dim, 
                          config.prediction_hidden_dim, 
                          config.prediction_output_dim).to(config.device)
        
        # create the memory bank, criterion and optimizer
        memory_bank = NNMemoryBankModule(size=(
            config.queue_size, 
            config.prediction_output_dim
            )).to(config.device)
        criterion = NTXentLoss(temperature=config.temperature, 
                               memory_bank_size=(config.queue_size, 
                                                 config.prediction_output_dim))
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            total_loss = 0
            for x0, x1, _ in dataloader:
                x0, x1 = x0.to(config.device), x1.to(config.device)
                _, z0, p0 = model(x0)
                _, z1, p1 = model(x1)
                z0 = memory_bank(z0, update=False) # update can be True for z0 xor z1
                z1 = memory_bank(z1, update=True)
                loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(dataloader)
            wandb.log({"loss": avg_loss}, step=epoch)
                
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_PATH, 
                             f"{view1_name}_{view2_name}_{epoch}_{config.reduce_factor}.pth"))
        torch.cuda.empty_cache()
        wandb.finish()