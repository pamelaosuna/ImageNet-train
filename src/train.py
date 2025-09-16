import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm

import numpy as np
import random


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

IMAGENET_TRAIN_SIZE = 1281167
IMAGENET_VAL_SIZE = 50000

from utils import get_imagenet_train_loader, get_imagenet_val_loader
from model import (
    get_model,
    get_optimizer,
    get_scheduler
    )

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc='Training', leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # gradient clipping
        scaler.step(optimizer)
        scaler.update()

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
        train_dir, batch_size=config["batch_size"], workers=config["workers"], debug=debug
        )
    val_loader = get_imagenet_val_loader(
        val_tar, val_labels_file, batch_size=config["batch_size"], workers=config["workers"], debug=debug
        )

    # Initialize Model
    model = get_model(config["model_name"], device)

    # Optimizer & loss
    optimizer = get_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler
    scheduler = get_scheduler(config, optimizer)

    scaler = torch.cuda.amp.GradScaler()

    # Training
    epochs = config["epochs"]
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, f"best_weights.pth")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
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
    argparser.add_argument('--debug', action='store_true',
        help='Whether to run in debug mode')
    argparser.add_argument('--model_name', type=str, default='alexnet',
        choices=['alexnet', 'resnet50', 'vit_b_16'],
        help='Model architecture to use.')
    argparser.add_argument('--batch_size', type=int, default=256,
        help='Batch size for training and validation')
    argparser.add_argument('--workers', type=int, default=4,
        help='Number of worker threads for data loading')
    args = argparser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_dir = os.path.join('./checkpoints', f"{args.model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Settings for AlexNet and ResNet50
    config = {
        "epochs": 90,
        "batch_size": args.batch_size,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "model_name": args.model_name,
        "timestamp": timestamp,
        "optimizer": "SGD",
        "scheduler": "StepLR",
        "step_size": 30,
        "gamma": 0.1,
        "workers": args.workers,
    }

    # Specific settings for ViT
    # epochs=300, warmup_epochs=10, base_lr=5e-4, batch_size=256, weight_decay=0.05
    if args.model_name == "vit_b_16":
        config["epochs"] = 300
        config["warmup_epochs"] = 10
        config["lr"] = 5e-4 * (args.batch_size / 1024)  # linear scaling with batch size
        config["weight_decay"] = 0.05
        config["scheduler"] = "LambdaLR"
        config["optimizer"] = "AdamW"

    if args.debug:
        config["epochs"] = 2
        config["batch_size"] = 16
        config["workers"] = 0

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    main(args.data_dir, save_dir=save_dir, config=config, debug=args.debug)


