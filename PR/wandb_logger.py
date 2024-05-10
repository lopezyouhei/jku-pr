import wandb

def initialize_wandb(project_name="NNCLR", config={}):
    wandb.init(project=project_name, 
               config=config)
    return wandb

def log_wandb(wandb, metrics, step):
    wandb.log(metrics, step=step)

def finish_wandb(wandb):
    wandb.finish()