#!/usr/bin/env python3
"""
Script to extract a subset of ImageNet classes from tar files or existing directories.
This creates the directory structure needed for the subset training script.
"""

import os
import shutil
import tarfile
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import json

def load_imagenet_class_mapping(mapping_file=None):
    """Load ImageNet index to class name mapping"""
    
    if mapping_file and os.path.exists(mapping_file):
        # Load from text file
        with open(mapping_file, 'r') as f:
            mapping = f.read()
            mapping = eval(mapping)  # Convert string to dict
        
        # print(mapping)
        
        return mapping
    else:
        raise ValueError("Mapping file not provided or does not exist.")

def extract_classes_from_tar(tar_pattern, output_dir, class_list, split_name="train", mapping=None):
    """Extract specific classes from ImageNet tar files"""
    
    print(f"Extracting {len(class_list)} classes from tar files: {tar_pattern}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all tar files
    tar_files = sorted(glob.glob(tar_pattern))
    if not tar_files:
        print(f"No tar files found matching: {tar_pattern}")
        return
    
    print(f"Found {len(tar_files)} tar files")
    
    extracted_classes = set()
    total_images = 0
    
    # Process each tar file
    for tar_path in tqdm(tar_files, desc="Processing tar files"):
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Get all members (files) in the tar
                members = tar.getmembers()
                
                for member in tqdm(members, desc=f"Processing {os.path.basename(tar_path)}", leave=False):
                    if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        if split_name == "train":
                            # Assuming format: class_name/image_name.jpg
                            path_parts = member.name.split('/')
                            if len(path_parts) >= 2:
                                class_name = path_parts[0]
                            else:
                                # Try to extract from filename (e.g., n01440764_10026.JPEG)
                                filename = os.path.basename(member.name)
                                class_name = filename.split('_')[0]
                        else:  # val split
                            # Extract class from .csl file with same name
                            filename = os.path.basename(member.name)
                            class_file = filename.rsplit('.', 1)[0] + '.cls'
                            class_name = None
                            for cls_member in members:
                                if cls_member.name.endswith(class_file):
                                    cls_file_data = tar.extractfile(cls_member)
                                    if cls_file_data:
                                        class_name = int(cls_file_data.read().decode('utf-8').strip())
                                    break
                            if class_name is None:
                                raise ValueError(f"Class file {class_file} not found in tar.")
                            else:
                                # Do mapping
                                if mapping and class_name in mapping:
                                    class_name = mapping[class_name]
                                else:
                                    raise ValueError(f"Class name {class_name} not found in mapping.")

                        
                        if class_name in class_list:
                            # Create class directory if it doesn't exist
                            class_dir = os.path.join(output_dir, class_name)
                            os.makedirs(class_dir, exist_ok=True)
                            
                            # Extract the file
                            output_path = os.path.join(class_dir, os.path.basename(member.name))
                            
                            # Extract file data
                            file_data = tar.extractfile(member)
                            if file_data:
                                with open(output_path, 'wb') as f:
                                    f.write(file_data.read())
                                
                                extracted_classes.add(class_name)
                                total_images += 1
                
        except Exception as e:
            print(f"Error processing {tar_path}: {e}")
            continue
    
    print(f"\nExtraction complete!")
    print(f"Extracted classes: {len(extracted_classes)} out of {len(class_list)}")
    print(f"Total images extracted: {total_images}")
    
    # Print classes that were found
    found_classes = extracted_classes.intersection(set(class_list))
    missing_classes = set(class_list) - extracted_classes
    
    if found_classes:
        print(f"\nFound classes ({len(found_classes)}):")
        for cls in sorted(found_classes):
            class_dir = os.path.join(output_dir, cls)
            num_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {cls}: {num_images} images")
    
    if missing_classes:
        print(f"\nMissing classes ({len(missing_classes)}):")
        for cls in sorted(missing_classes):
            print(f"  {cls}")

def copy_classes_from_directory(source_dir, output_dir, class_list):
    """Copy specific classes from an existing ImageNet directory structure"""
    
    print(f"Copying {len(class_list)} classes from: {source_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    copied_classes = set()
    total_images = 0
    
    for class_name in tqdm(class_list, desc="Copying classes"):
        source_class_dir = os.path.join(source_dir, class_name)
        
        if os.path.exists(source_class_dir) and os.path.isdir(source_class_dir):
            output_class_dir = os.path.join(output_dir, class_name)
            
            # Copy the entire class directory
            shutil.copytree(source_class_dir, output_class_dir, dirs_exist_ok=True)
            
            # Count images
            num_images = len([f for f in os.listdir(output_class_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            copied_classes.add(class_name)
            total_images += num_images
            
            print(f"Copied {class_name}: {num_images} images")
        else:
            print(f"Warning: Class directory not found: {source_class_dir}")
    
    print(f"\nCopy complete!")
    print(f"Copied classes: {len(copied_classes)} out of {len(class_list)}")
    print(f"Total images copied: {total_images}")
    
    missing_classes = set(class_list) - copied_classes
    if missing_classes:
        print(f"\nMissing classes: {sorted(missing_classes)}")

def load_class_list(file_path):
    """Load class list from file"""
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def main():
    parser = argparse.ArgumentParser(description='Extract subset of ImageNet classes')
    
    # Input/output arguments
    parser.add_argument('--source', required=True,
                       help='Source: tar file pattern (e.g., "/path/to/*.tar") or directory path')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for extracted classes')
    parser.add_argument('--class-list', 
                       help='Text file with class names (one per line)')
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                       help='Dataset split (train uses folder names, val uses .cls files)')
    parser.add_argument('--imagenet-mapping', default=None,
                        help='Text file with ImageNet index to class name mapping')
    
    # Options
    parser.add_argument('--mode', choices=['tar', 'directory'], default='auto',
                       help='Extraction mode: tar files or existing directory structure')
    
    args = parser.parse_args()
    
    # Load class list
    if not args.class_list:
        print("Error: --class-list is required (or use --create-default-list)")
        return
    
    class_list = load_class_list(args.class_list)
    print(f"Loaded {len(class_list)} classes from {args.class_list}")

    mapping = None
    if args.split == 'val':
        mapping = load_imagenet_class_mapping(args.imagenet_mapping)
    
    # Auto-detect mode if not specified
    if args.mode == 'auto':
        if '*' in args.source or args.source.endswith('.tar'):
            args.mode = 'tar'
        else:
            args.mode = 'directory'
    
    print(f"Using mode: {args.mode}")
    
    # Extract classes
    if args.mode == 'tar':
        extract_classes_from_tar(args.source, args.output_dir, class_list, args.split, mapping)
    else:
        copy_classes_from_directory(args.source, args.output_dir, class_list)
    
    print("\nDone! You can now use the extracted data with the subset training script:")
    print(f"python train_subset.py --train-data {args.output_dir} --val-data <val_dir> --class-list {args.class_list}")

if __name__ == "__main__":
    main()