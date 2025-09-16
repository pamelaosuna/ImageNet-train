import os
import argparse
import json
from datetime import datetime

import numpy as np
import random


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from utils import get_imagenet_train_loader, get_imagenet_val_loader

IMAGENET_TRAIN_SIZE = 1281167
IMAGENET_VAL_SIZE = 50000

from model import get_model

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc='Training', leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # gradient clipping
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / IMAGENET_TRAIN_SIZE

def validate(model, loader, criterion, device):
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation', leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()

    acc = correct / IMAGENET_VAL_SIZE
    return val_loss / IMAGENET_VAL_SIZE, acc


def main(data_dir, save_dir, config, debug):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")

    train_dir = os.path.join(data_dir, 'ILSVRC2012_img_train')
    val_tar = os.path.join(data_dir, 'ILSVRC2012_img_val.tar')
    val_labels_file = os.path.join(data_dir, 'ILSVRC2012_validation_ground_truth.txt')

    train_loader = get_imagenet_train_loader(
        train_dir, batch_size=config["batch_size"], workers=config["workers"], debug=debug, seed=config["seed"]
        )
    val_loader = get_imagenet_val_loader(
        val_tar, val_labels_file, batch_size=config["batch_size"], workers=config["workers"], debug=debug
        )

    set_seed(config["seed"])

    # Initialize Model
    model = get_model(config["model_name"], device)

    # Optimizer & loss
    # TODO: set optimizer based on config and dependent on model, the current is for AlexNet
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config["lr"], 
        momentum=config["momentum"], 
        weight_decay=config["weight_decay"]
        )
    criterion = nn.CrossEntropyLoss()

    # LR scheduler
    # TODO: set scheduler based on config
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # Training
    epochs = config["epochs"]
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, f"best_weights.pth")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} "
                f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        # Save model only if val_loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at epoch {epoch+1} with improved val_loss: {val_loss:.4f}")

        if epoch % 19 == 0 or epoch == epochs - 1:
            intermediate_model_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), intermediate_model_path)
            print(f"Intermediate model saved at epoch {epoch+1}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train AlexNet on ImageNet from scratch.")
    argparser.add_argument('--data_dir', type=str, required=True, 
        help='Path to ImageNet data directory')
    argparser.add_argument('--model_subdir', type=str)
    argparser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility')
    argparser.add_argument('--debug', action='store_true',
        help='Whether to run in debug mode')
    argparser.add_argument('--model_name', type=str, default='alexnet',
        choices=['alexnet', 'resnet50', 'vit_b_16'],
        help='Model architecture to use.')
    argparser.add_argument('--epochs', type=int, default=90,
        help='Number of training epochs')
    argparser.add_argument('--lr', type=float, default=0.01,
        help='Learning rate')
    argparser.add_argument('--batch_size', type=int, default=256,
        help='Batch size for training and validation')
    args = argparser.parse_args()

    save_dir = os.path.join('./checkpoints', args.model_subdir)
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "seed": args.seed,
        "initialization": "kaiming_normal",
        "model_name": args.model_name,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "optimizer": "SGD",
        "scheduler": "StepLR",
        "step_size": 30,
        "gamma": 0.1,
        "workers": 8
    }

    if args.debug:
        config["epochs"] = 2
        config["batch_size"] = 16
        config["workers"] = 0

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    main(args.data_dir, save_dir=save_dir, config=config, debug=args.debug)


