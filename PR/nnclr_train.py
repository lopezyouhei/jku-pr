import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from lightly.loss import NTXentLoss
from lightly.models.modules import (NNCLRProjectionHead,
                                    NNCLRPredictionHead,
                                    NNMemoryBankModule)

from PR.wandb_logger import *

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 4096
LR = 0.4 # 0.3
TEMPERATURE = 0.15
QUEUE_SIZE = 98304
WEIGHT_DECAY = 1e-6
PROJECT_HIDDEN_DIM = 2048
PROJECT_OUTPUT_DIM = 256
PREDICTION_HIDDEN_DIM = 4096
PREDICTION_OUTPUT_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MAEDataset(torch.utils.data.Dataset):
    def __init__(self, view1_path, view2_path):

        self.data1 = torch.load(view1_path)
        self.data2 = torch.load(view2_path)
        assert self.data1.shape == self.data2.shape, "view1 and view2 must have the same shape"
        self.length = self.data1.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]
    
class NNCLRHead(nn.Module):
    def __init__(self, 
                 project_hidden_dim=PROJECT_HIDDEN_DIM, 
                 project_output_dim=PROJECT_OUTPUT_DIM, 
                 predict_hidden_dim=PREDICTION_HIDDEN_DIM, 
                 predict_output_dim=PREDICTION_OUTPUT_DIM):
        super().__init__()

        self.projection_head = NNCLRProjectionHead(1024, # input_dim as provided by supervisors
                                                   project_hidden_dim, # hidden_dim 
                                                   project_output_dim) # output_dim
        self.prediction_head = NNCLRPredictionHead(project_output_dim, # input_dim 
                                                   predict_hidden_dim, # hidden_dim
                                                   predict_output_dim) # output_dim

    def forward(self, x):
        z = self.projection_head(x)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

# In case you want to change the model architecture, add a new input argument 
# to the function and pass it to the model initialization, if the same model 
# should be used the variable should be set to None for example.
def train(view1_path, view2_path):
    view1 = view1_path.split("/")[-1]
    view2 = view2_path.split("/")[-1]

    # initialize wandb
    wandb = initialize_wandb("NNCLR", 
                             {
                                 "view1": view1,
                                 "view2": view2,
                                 "projection_hidden_dim": PROJECT_HIDDEN_DIM,
                                 "projection_output_dim": PROJECT_OUTPUT_DIM,
                                 "prediction_hidden_dim": PREDICTION_HIDDEN_DIM,
                                 "prediction_output_dim": PREDICTION_OUTPUT_DIM,
                                 "epochs": EPOCHS, 
                                 "batch_size": BATCH_SIZE, 
                                 "lr": LR, 
                                 "temperature": TEMPERATURE, 
                                 "queue_size": QUEUE_SIZE, 
                                 "weight_decay": WEIGHT_DECAY,
                                 }
                                 )
    
    # create a dataset from the view1 and view2 paths
    dataset = MAEDataset(view1_path, view2_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create the NNCLR model
    model = NNCLRHead().to(DEVICE)

    # create the criterion and optimizer
    criterion = NTXentLoss(temperature=TEMPERATURE, 
                           memory_bank_size=(QUEUE_SIZE, PREDICTION_OUTPUT_DIM)
                           )
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=LR, 
                                weight_decay=WEIGHT_DECAY)
    
    # create the memory bank
    memory_bank = NNMemoryBankModule(size=(QUEUE_SIZE, PREDICTION_OUTPUT_DIM))
    memory_bank = memory_bank.to(DEVICE)

    best_loss = math.inf

    # start the training loop
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x0, x1 in dataloader:
            x0, x1 = x0.to(DEVICE), x1.to(DEVICE)
            z0, p0 = model(x0)
            z1, p1 = model(x1)
            z0 = memory_bank(z0, update=False) # update can be True for z0 xor z1
            z1 = memory_bank(z1, update=True)
            loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)

        # log the loss to wandb
        log_wandb(wandb, {"loss": avg_loss}, epoch)

        # save the model if the loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            view1_name = view1.split(".")[0]
            view2_name = view2.split(".")[0]
            torch.save(model.state_dict(), f"best_model_{view1_name}_{view2_name}.pth")

    torch.cuda.empty_cache()
    finish_wandb(wandb)
    

