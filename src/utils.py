import os
import io

import torch
from torchvision import transforms
from PIL import Image

import webdataset as wds

def build_class_mapping(tar_dir):
    wnids = sorted([f.replace(".tar", "") for f in os.listdir(tar_dir) if f.endswith(".tar")])
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    return class_to_idx

def get_imagenet_train_loader(tar_dir, batch_size=256, workers=8, debug=False):
    class_to_idx = build_class_mapping(tar_dir)
    shards = [os.path.join(tar_dir, fname) for fname in os.listdir(tar_dir) if fname.endswith(".tar")]

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def preprocess(sample):
        img_bytes, key = sample   # key is like "n01440764_18"
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = transform(img)

        wnid = key.split("_")[0]  # get "n01440764"
        target = class_to_idx[wnid]
        return img, target

    if debug:
        dataset = (
            wds.WebDataset(shards, handler=wds.warn_and_continue, shardshuffle=False)
            .to_tuple("jpeg", "__key__")
            .map(preprocess)
            .slice(1000)
        )
    else:
        dataset = (
            wds.WebDataset(shards, handler=wds.warn_and_continue, shardshuffle=1000)
            .to_tuple("jpeg", "__key__")
            .shuffle(1000)
            .map(preprocess)
        )

    loader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        num_workers=workers,
        pin_memory=False,
    )
    return loader


def load_val_labels(val_labels_file):
    with open(val_labels_file) as f:
        labels = [int(x.strip()) - 1 for x in f.readlines()]  # convert to 0-based
    return labels


def get_imagenet_val_loader(val_tar, val_labels_file, batch_size=256, workers=8, debug=False):
    val_labels = load_val_labels(val_labels_file)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def preprocess(sample):
        img_bytes, key = sample        # tuple unpacking
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = transform(img)

        # get index from key, e.g. ILSVRC2012_val_00000001 → 0-based index
        idx = int(key.split("_")[-1]) - 1
        target = val_labels[idx]
        return img, target

    # # debug mode
    # dataset = wds.WebDataset(val_tar, handler=wds.ignore_and_continue).decode()
    # for sample in dataset:
    #     print(f'DEBUG -- sample keys: {list(sample.keys())}')
    #     break

    if debug:
        dataset = (
            wds.WebDataset(val_tar, handler=wds.warn_and_continue, shardshuffle=False)
            .to_tuple("jpeg;JPEG", "__key__")
            .map(preprocess)
            .slice(500)
        )
    else:
        dataset = (
            wds.WebDataset(val_tar, handler=wds.warn_and_continue, shardshuffle=False)
            .to_tuple("jpeg;JPEG", "__key__")
            .map(preprocess)
        )
        
    loader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        num_workers=0, # single tar file -> single worker
        pin_memory=False,
    )
    return loader