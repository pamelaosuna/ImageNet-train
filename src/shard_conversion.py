import os
import tarfile
import scipy.io
import webdataset as wds
from tqdm import tqdm

# -------- CONFIG --------
train_dir = "data_ImageNet/ILSVRC2012_img_train"  # folder containing 1000 wnid tars
val_tar   = "data_ImageNet/ILSVRC2012_img_val.tar"
val_labels_txt = "data_ImageNet/ILSVRC2012_validation_ground_truth.txt"
devkit_meta = data_ImageNet/ILSVRC2012_devkit_t12/data/meta.mat"

out_dir   = "data_ImageNet/imagenet-wds"
samples_per_shard = 10000
# ------------------------

os.makedirs(out_dir, exist_ok=True)

# -------- LOAD WNID->IDX MAPPING --------
print("Loading wnid -> idx mapping from devkit...")
meta = scipy.io.loadmat(devkit_meta, squeeze_me=True)["synsets"]
wnid_to_idx = {row[1]: int(row[0]) - 1 for row in meta if int(row[0]) <= 1000}

def write_shard(samples, split, shard_id):
    shard_name = f"{out_dir}/{split}-{shard_id:05d}.tar"
    with wds.TarWriter(shard_name) as sink:
        for key, image_bytes, label in samples:
            sink.write({
                "__key__": key,
                "jpg": image_bytes,
                "cls": str(label).encode("utf-8"),
            })


# -------- TRAIN CONVERSION --------
print("Converting training set...")
train_tars = sorted(os.listdir(train_dir))
samples, shard_id = [], 0

for tar_name in tqdm(train_tars):
    wnid = os.path.splitext(tar_name)[0]  # e.g. n01440764
    if wnid not in wnid_to_idx:
        continue
    label = wnid_to_idx[wnid]

    with tarfile.open(os.path.join(train_dir, tar_name), "r") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            img_bytes = tf.extractfile(m).read()
            key = os.path.splitext(os.path.basename(m.name))[0]
            samples.append((key, img_bytes, label))

            if len(samples) >= samples_per_shard:
                write_shard(samples, "train", shard_id)
                shard_id += 1
                samples = []

if samples:
    write_shard(samples, "train", shard_id)


# -------- VAL CONVERSION --------
print("Converting validation set...")
with open(val_labels_txt) as f:
    val_labels = [int(line.strip()) for line in f]

with tarfile.open(val_tar, "r") as tf:
    members = sorted([m for m in tf.getmembers() if m.isfile()],
                     key=lambda x: x.name)
    samples, shard_id = [], 0
    for idx, m in enumerate(tqdm(members)):
        label = val_labels[idx] - 1 # to zero-based
        img_bytes = tf.extractfile(m).read()
        key = os.path.splitext(os.path.basename(m.name))[0]
        samples.append((key, img_bytes, label))

        if len(samples) >= samples_per_shard:
            write_shard(samples, "val", shard_id)
            shard_id += 1
            samples = []

    if samples:
        write_shard(samples, "val", shard_id)
