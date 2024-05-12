import argparse
from lightly.loss import NTXentLoss
from lightly.models.modules import NNMemoryBankModule
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

def train(user_tags, args, random_views=True):
    with wandb.init(project="NNCLR", tags=user_tags) as run:
        config = run.config

        config.batch_size = args.batch_size
        config.lr = args.lr
        config.temperature = args.temperature
        config.queue_size = args.queue_size
        config.weight_decay = args.weight_decay
        config.reduce_factor = args.reduce_factor

        config.random_views = config.get("random_views", random_views)
        config.epochs = config.get("epochs", round(EPOCHS/config.reduce_factor))
        config.projection_hidden_dim = config.get("project_hidden_dim", PROJECT_HIDDEN_DIM)
        config.projection_output_dim = config.get("project_output_dim", PROJECT_OUTPUT_DIM)
        config.prediction_hidden_dim = config.get("prediction_hidden_dim", PREDICTION_HIDDEN_DIM)
        config.prediction_output_dim = config.get("prediction_output_dim", PREDICTION_OUTPUT_DIM)
        config.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # create a dataset
        dataset = None
        view1_path, view2_path = generate_pair(config.random_views, 
                                               DATA_PATH, 
                                               MODEL_PATH)
        
        view1_name = view1_path.split("/")[-1].split(".")[0]
        view2_name = view2_path.split("/")[-1].split(".")[0]

        config.view1 = config.get("view1", view1_name)
        config.view2 = config.get("view2", view2_name)

        dataset = MAEDataset(view1_path, 
                             view2_path,
                             labels_path,
                             config.reduce_factor)

        # create the dataloader
        dataloader = DataLoader(dataset, 
                                batch_size=config.batch_size, 
                                shuffle=True)

        # create the NNCLR model
        model = NNCLRHead(config.projection_hidden_dim, 
                          config.projection_output_dim, 
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

            red_fac_name = str(config.reduce_factor).replace(".", "-")    
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_PATH, 
                             f"{view1_name}_{view2_name}_{epoch}_{red_fac_name}.pth"))
        torch.cuda.empty_cache()
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--queue_size", type=int, default=QUEUE_SIZE, help="Queue size for memory bank")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for NTXent loss")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--reduce_factor", type=float, default=1.0, help="Reduce factor for class number reduction")
    args = parser.parse_args()

    tags = ["baseline_HPO"]

    for _ in range(args.runs):
        train(tags, args, random_views=False)