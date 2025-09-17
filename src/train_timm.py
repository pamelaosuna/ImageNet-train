import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import webdataset as wds
import timm
from timm.data import create_transform
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import AverageMeter, accuracy
import argparse
from pathlib import Path
import time
import json

from torchvision.models import alexnet

def create_webdataset_loader(tar_files, batch_size, transform, shuffle=True, num_workers=4):
    """Create WebDataset loader from tar files"""
    
    def identity(x):
        return x
    
    # Create dataset from tar files
    dataset = (
        wds.WebDataset(tar_files, shardshuffle=shuffle)
        .shuffle(1000 if shuffle else 0)  # Shuffle samples within shards
        .decode("pil")  # Decode images as PIL
        .to_tuple("jpg;png", "cls")  # Extract image and class
        .map_tuple(transform, identity)  # Apply transforms to image only
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset.batched(batch_size),
        batch_size=None,  # Batching handled by webdataset
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

def create_transforms():
    """Create ImageNet transforms for training and validation"""
    
    # Training transforms (with augmentation)
    train_transform = create_transform(
        input_size=(3, 224, 224),
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',  # RandAugment
        re_prob=0.25,  # Random erasing
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225],   # ImageNet std
    )
    
    # Validation transforms (no augmentation)
    val_transform = create_transform(
        input_size=(3, 224, 224),
        is_training=False,
        interpolation='bicubic',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx:>4d}] '
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc@1: {top1.val:.2f} ({top1.avg:.2f}) '
                  f'Acc@5: {top5.val:.2f} ({top5.avg:.2f})')
    
    return losses.avg, top1.avg, top5.avg

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Metrics
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
    
    print(f'Validation: Loss: {losses.avg:.4f} '
          f'Acc@1: {top1.avg:.2f} Acc@5: {top5.avg:.2f}')
    
    return losses.avg, top1.avg, top5.avg

def main():
    parser = argparse.ArgumentParser(description='Train network on ImageNet with WebDataset')
    
    # Data arguments
    parser.add_argument('--train-data', required=True, 
                       help='Path to training tar files (glob pattern)')
    parser.add_argument('--val-data', required=True,
                       help='Path to validation tar files (glob pattern)')
    
    # Model arguments
    parser.add_argument('--model', default='alexnet', 
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=1000,
                       help='Number of classes')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=90,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', default='cuda',
                       help='Device to use for training')
    parser.add_argument('--output-dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    # Create datasets
    print('Creating datasets...')
    train_loader = create_webdataset_loader(
        args.train_data, 
        args.batch_size, 
        train_transform, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = create_webdataset_loader(
        args.val_data,
        args.batch_size,
        val_transform,
        shuffle=False,
        num_workers=min(args.num_workers, 4)  # fewer workers for validation
    )
    
    # Create model
    print(f'Creating model: {args.model}')
    if args.model == 'alexnet':
        # not available in timm, use torchvision
        model = alexnet(
            pretrained=args.pretrained, 
            num_classes=args.num_classes
        )
    else:
        model = timm.create_model(
            args.model, 
            pretrained=args.pretrained,
            num_classes=args.num_classes
        )
    model = model.to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Scheduler (StepLR as used in original AlexNet)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    best_acc1 = 0
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        checkpoint = {
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f'epoch_{epoch+1}.pth'))
        
        print(f'Best Acc@1: {best_acc1:.2f}')

if __name__ == '__main__':
    main()
    # TODO: have a config file and save it with the checkpoints, have a subdir for each run