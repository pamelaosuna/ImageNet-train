import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import (
    alexnet,
    resnet50,
    vit_b_16
)

def get_model(model_name, device):
    if model_name == "alexnet":
        model = alexnet(num_classes=1000, weights=None)
        # model.apply(init_weights)  # Custom init
    elif model_name == "resnet50":
        model = resnet50(num_classes=1000, weights=None)
    elif model_name == "vit_b_16":
        model = vit_b_16(num_classes=1000, weights=None)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    model = model.to(device)

    return model

def lr_lambda(current_epoch, config):
    if current_epoch < config["warmup_epochs"]:
        return float(current_epoch) / float(max(1, config["warmup_epochs"]))
    progress = float(current_epoch - config["warmup_epochs"]) / float(max(1, config["epochs"] - config["warmup_epochs"]))
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

def get_optimizer(model, config):
    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config["lr"], 
            momentum=config["momentum"], 
            weight_decay=config["weight_decay"]
            )
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
            )
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented.")
        
    return optimizer

def get_scheduler(config, optimizer):
    if config["scheduler"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config["step_size"], gamma=config["gamma"]
            )
    elif config["scheduler"] == "LambdaLR":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, config)
            )
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented.")

    return scheduler