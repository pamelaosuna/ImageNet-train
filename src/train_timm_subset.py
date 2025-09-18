import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import timm
from timm.data import create_transform
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import AverageMeter, accuracy, reduce_tensor
import argparse
from pathlib import Path
import time
import json
import math

from torchvision.models import alexnet

class ImageNetSubsetDataset(datasets.ImageFolder):
    """Custom ImageFolder that works with a subset of ImageNet classes"""
    
    def __init__(self, root, class_list, transform=None, **kwargs):
        # First, create the full ImageFolder to get all classes
        temp_dataset = datasets.ImageFolder(root)
        
        # Create mapping from class names to indices
        self.original_class_to_idx = temp_dataset.class_to_idx
        self.selected_classes = class_list
        
        # Verify that all selected classes exist
        missing_classes = [cls for cls in class_list if cls not in self.original_class_to_idx]
        if missing_classes:
            print(f"Warning: These classes not found in dataset: {missing_classes}")
            self.selected_classes = [cls for cls in class_list if cls in self.original_class_to_idx]
        
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Using {len(self.selected_classes)} classes out of {len(class_list)} requested")
        elif not dist.is_initialized():
            print(f"Using {len(self.selected_classes)} classes out of {len(class_list)} requested")
        
        # Create new class mapping (0 to num_selected_classes-1)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Initialize parent with transform
        super().__init__(root, transform=transform, **kwargs)
        
        # Filter samples to only include selected classes
        self.samples = self._filter_samples()
        
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Dataset created with {len(self.samples)} samples from {len(self.selected_classes)} classes")
        elif not dist.is_initialized():
            print(f"Dataset created with {len(self.samples)} samples from {len(self.selected_classes)} classes")
    
    def _filter_samples(self):
        """Filter samples to only include selected classes"""
        filtered_samples = []
        
        for path, original_label in self.samples:
            # Get class name from path
            class_name = os.path.basename(os.path.dirname(path))
            
            if class_name in self.selected_classes:
                # Map to new label (0 to num_selected_classes-1)
                new_label = self.class_to_idx[class_name]
                filtered_samples.append((path, new_label))
        
        return filtered_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

def load_class_list_from_file(file_path):
    """Load class list from a text file (one class per line)"""
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

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

def setup_distributed(rank, world_size, port=12355):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_epoch(model, loader, criterion, optimizer, device, epoch, clipping, mixup_fn, distributed=False):
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
            
            # For distributed training, we need to reduce metrics across processes
            if distributed:
                loss = reduce_tensor(loss, dist.get_world_size())
                acc1 = reduce_tensor(acc1, dist.get_world_size())
                acc5 = reduce_tensor(acc5, dist.get_world_size())
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        else:
            # For distributed training, reduce loss
            if distributed:
                loss = reduce_tensor(loss, dist.get_world_size())
            
            losses.update(loss.item(), images.size(0))
            # For mixup, we can't compute exact accuracy
            top1.update(0.0, images.size(0))
            top5.update(0.0, images.size(0))
        
        # Only print from rank 0 in distributed training
        if batch_idx % 100 == 0 and (not distributed or dist.get_rank() == 0):
            print(f'Epoch: {epoch + 1} [{batch_idx:>4d}] '
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc@1: {top1.val:.2f} ({top1.avg:.2f}) '
                  f'Acc@5: {top5.val:.2f} ({top5.avg:.2f})')
    
    return losses.avg, top1.avg, top5.avg

def validate(model, loader, criterion, device, distributed=False):
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
            
            # For distributed training, reduce metrics across processes
            if distributed:
                loss = reduce_tensor(loss, dist.get_world_size())
                acc1 = reduce_tensor(acc1, dist.get_world_size())
                acc5 = reduce_tensor(acc5, dist.get_world_size())
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
    
    # Only print from rank 0 in distributed training
    if not distributed or dist.get_rank() == 0:
        print(f'Validation: Loss: {losses.avg:.4f} '
              f'Acc@1: {top1.avg:.2f} Acc@5: {top5.avg:.2f}')
    
    return losses.avg, top1.avg, top5.avg

def create_config(args):
    """Create configuration dictionary from arguments"""
    config = {
        'data': {
            'train_data': args.train_data,
            'val_data': args.val_data,
            'class_list': args.class_list,
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
            'output_dir': args.output_dir,
            'distributed': args.distributed,
            'world_size': args.world_size
        }
    }
    return config

def run_training(rank, world_size, args):
    """Main training function for distributed training"""
    # Setup distributed training if using multiple GPUs
    distributed = world_size > 1
    if distributed:
        setup_distributed(rank, world_size, args.port)
    
    # Device setup
    if distributed:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f'Using device: {device}')
        if distributed:
            print(f'Distributed training on {world_size} GPUs')
    
    # Load class list
    if rank == 0:
        print(f"Loading class list from: {args.class_list}")
    class_names = load_class_list_from_file(args.class_list)
    args.num_classes = len(class_names)
    if rank == 0:
        print(f"Training on {args.num_classes} classes")

    # Create output directory (only on rank 0)
    if rank == 0:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join(args.output_dir, f'{args.model}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        config = create_config(args)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    else:
        output_dir = None
    
    # Broadcast output_dir to all processes
    if distributed:
        output_dir_list = [output_dir]
        dist.broadcast_object_list(output_dir_list, src=0)
        output_dir = output_dir_list[0]

    # Create transforms
    train_transform, val_transform = create_transforms(vit=(args.model.startswith('vit')))
    
    # Create datasets
    if rank == 0:
        print('Creating datasets...')
    train_dataset = ImageNetSubsetDataset(
        args.train_data, 
        class_names, 
        transform=train_transform
    )
    
    val_dataset = ImageNetSubsetDataset(
        args.val_data,
        class_names,
        transform=val_transform
    )

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using DistributedSampler
        sampler=train_sampler,
        num_workers=args.num_workers,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    
    # Create model
    if rank == 0:
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
    
    # Wrap model for distributed training
    if distributed:
        model = DDP(model, device_ids=[rank])
    elif torch.cuda.device_count() > 1 and not distributed:
        # Use DataParallel if multiple GPUs but not using distributed
        model = nn.DataParallel(model)
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'Model parameters: {param_count:.2f}M')
    
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
        if rank == 0:
            print(f'Using Mixup (alpha={args.mixup}) and CutMix (alpha={args.cutmix})')
    
    # Optimizer - scale learning rate for distributed training
    lr = args.lr * world_size if distributed else args.lr
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
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
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=10,
            max_epochs=args.epochs,
            base_lr=lr,
            min_lr=1e-6
        )
    else:
        raise NotImplementedError(f'Scheduler not defined: {args.scheduler}')
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc1 = 0

    if args.resume and os.path.isfile(args.resume):
        if rank == 0:
            print(f'Loading checkpoint from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle DDP state dict loading
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint.get('best_acc1', 0)
    
    if args.resume is None and rank == 0:
        # save network before training (only on rank 0)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': 0,
            'model': args.model,
            'state_dict': model_to_save.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, os.path.join(output_dir, 'initial.pth'))

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f'\nEpoch {epoch+1}/{args.epochs}')
            print('-' * 50)

        # Cosine annealing scheduler step at start of epoch
        if args.scheduler == 'cosine':
            scheduler.step(epoch)
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            clipping=(args.model.startswith('vit')),
            mixup_fn=mixup_fn,
            distributed=distributed
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, distributed=distributed
        )
        
        # Update scheduler
        if args.scheduler == 'steplr':
            scheduler.step()
        
        # Save checkpoint (only from rank 0)
        if rank == 0:
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)
            
            # Get the actual model (unwrap DDP if necessary)
            model_to_save = model.module if hasattr(model, 'module') else model
            
            checkpoint = {
                'epoch': epoch,
                'model': args.model,
                'state_dict': model_to_save.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            
            # Save best checkpoint
            if is_best:
                torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))

            # Save every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
                torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch+1}.pth'))
            
            print(f'Best Acc@1: {best_acc1:.2f}')
    
    # Clean up distributed training
    if distributed:
        cleanup_distributed()

def main(args):
    # Auto-detect world size if not specified
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
    
    # Run training
    if args.distributed and args.world_size > 1:
        if args.world_size > torch.cuda.device_count():
            print(f"Error: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available")
            return
        print(f'Starting distributed training on {args.world_size} GPUs')
        mp.spawn(run_training, args=(args.world_size, args), nprocs=args.world_size)
    else:
        # Single GPU training
        run_training(0, 1, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network on ImageNet subset with multi-GPU support')
    
    # Data arguments
    parser.add_argument('--train-data', required=True, 
                       help='Path to training data directory (ImageFolder format)')
    parser.add_argument('--val-data', required=True,
                       help='Path to validation data directory (ImageFolder format)')
    parser.add_argument('--class-list', required=True,
                       help='Path to text file with class names (one per line).')
    
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
                       help='Batch size per GPU')
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
    parser.add_argument('--mixup', type=float, default=0.8,
                       help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=1.0,
                       help='CutMix alpha')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--world-size', type=int, default=-1,
                       help='Number of GPUs to use for distributed training')
    parser.add_argument('--port', type=int, default=12355,
                       help='Port for distributed training')
            
    # System arguments
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers (set to 0 to avoid multiprocessing issues)')
    parser.add_argument('--output-dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()

    main(args)