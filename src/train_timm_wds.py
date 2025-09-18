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
        pin_memory=False
    )
    
    return loader

def create_transforms(vit=False):
    """Create ImageNet transforms for training and validation"""
    
    if not vit:
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
    else:
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
            color_jitter=0.4, # Additional augmentations for ViT
            mixup=0.8,          # Strong mixup for ViT
            cutmix=1.0,         # CutMix augmentation
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode='batch',
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

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def train_epoch(model, loader, criterion, optimizer, device, epoch, clipping, mixup_fn):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup/cutmix if provided
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for ViT stability
        if clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        # Metrics
        if mixup_fn is None:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        else:
            losses.update(loss.item(), images.size(0))
            # For mixup, we can't compute exact accuracy
            top1.update(0.0, images.size(0))
            top5.update(0.0, images.size(0))
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch + 1} [{batch_idx:>4d}] '
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

def create_config(args):
    """Create configuration dictionary from arguments"""
    config = {
        'data': {
            'train_data': args.train_data,
            'val_data': args.val_data
        },
        'model': {
            'name': args.model,
            'num_classes': args.num_classes,
            'pretrained': args.pretrained,
            'resume': args.resume,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'optimizer': {
                'type': args.optimizer,
                'weight_decay': args.weight_decay,
                'momentum': args.momentum,
                'betas': args.betas,
            }
        },
        'scheduler': {
            'type': args.scheduler,
            'step_size': args.step_size,
            'gamma': args.gamma
        },
        'system': {
            'num_workers': args.num_workers,
            'output_dir': args.output_dir
        }
    }
    return config

def main(args):
    # Create output directory
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Save config
    config = create_config(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Create transforms
    train_transform, val_transform = create_transforms(vit=(args.model.startswith('vit')))
    
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
    elif args.model == 'resnet50':
        model = timm.create_model(
            args.model, 
            pretrained=args.pretrained,
            num_classes=args.num_classes
        )
    elif args.model.startswith('vit'):
        model = timm.create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
    model = model.to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # Loss function
    if args.label_smoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    # Mixup/CutMix setup
    mixup_fn = None
    if args.use_mixup:
        from timm.data import Mixup
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=0.1,
            num_classes=args.num_classes
        )
        print(f'Using Mixup (alpha={args.mixup}) and CutMix (alpha={args.cutmix})')
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=args.betas
        )
    else:
        raise NotImplementedError(f'Optimizer not defined: {args.optimizer}')
    
    if args.scheduler == 'steplr':
        # Scheduler (StepLR as used in original AlexNet)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        # # Cosine annealing scheduler
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=args.epochs,
        #     eta_min=1e-6
        # )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=10,
            max_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=1e-6
        )
    else:
        raise NotImplementedError(f'Scheduler not defined: {args.scheduler}')
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc1 = 0

    if args.resume and os.path.isfile(args.resume):
        print(f'Loading checkpoint from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint.get('best_acc1', 0)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)

        # Cosine annealing scheduler step at start of epoch
        if args.scheduler == 'cosine':
            scheduler.step(epoch)
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            clipping=(args.model.startswith('vit')),
            mixup_fn=mixup_fn
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if args.scheduler == 'steplr':
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
        
        # # Save latest checkpoint
        # torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch+1}.pth'))
        
        print(f'Best Acc@1: {best_acc1:.2f}')

if __name__ == '__main__':
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
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
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
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='AdamW betas')
    parser.add_argument('--optimizer', default='sgd',
                       help='Optimizer (sgd or adamw)')
    parser.add_argument('--scheduler', default='steplr',
                        help='Learning rate scheduler (steplr or cosine)')
    parser.add_argument('--step-size', type=int, default=30,
                       help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='Use label smoothing in loss function')
    parser.add_argument('--use_mixup', action='store_true',
                        help='Use mixup augmentation')
            
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output-dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()

    main(args)
    # TODO: test ViT