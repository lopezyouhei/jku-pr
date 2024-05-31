from lightly.loss import NTXentLoss
from lightly.models.modules import (NNCLRProjectionHead,
                                    NNCLRPredictionHead,
                                    NNMemoryBankModule)
import os
import sys
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config.model_config import PROJECT_HIDDEN_DIM, PROJECT_OUTPUT_DIM, PREDICTION_HIDDEN_DIM, PREDICTION_OUTPUT_DIM



class NNCLRHead(nn.Module):
    def __init__(self, 
                 project_hidden_dim=PROJECT_HIDDEN_DIM, 
                 project_output_dim=PROJECT_OUTPUT_DIM, 
                 predict_hidden_dim=PREDICTION_HIDDEN_DIM, 
                 predict_output_dim=PREDICTION_OUTPUT_DIM):
        super().__init__()

        self.projection_head = NNCLRProjectionHead(1024, # input_dim 
                                                   project_hidden_dim, # hidden_dim 
                                                   project_output_dim) # output_dim
        self.prediction_head = NNCLRPredictionHead(project_output_dim, # input_dim 
                                                   predict_hidden_dim, # hidden_dim
                                                   predict_output_dim) # output_dim
        
        self.proj_activation = None
        self.hook = self.projection_head.layers[2].register_forward_hook(
            self.capture_activations
            )
    
    def capture_activations(self, module, input, output):
        self.proj_activation = output

    def forward(self, x):
        z = self.projection_head(x)
        p = self.prediction_head(z)
        z = z.detach()
        return self.proj_activation, z, p
    
    def remove_hook(self):
        self.hook.remove()