# This script create train.txt and val.txt file in a destination folder
# with (absolute-file-path,label) data per line
# Author: Soumen Sardar
# Date: 2026-02-18
# ####################################################################### 

import os
import pathlib
import shutil
import random

import argparse
import os
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Split dataset into train(80%) and val(20%) "
                                     "sets and create corresponding txt files for DINO")

    parser.add_argument(
        "input_dir",
        type=str,
        default="/data/data_tip_bag_pool_png",
        help="Path to the input directory"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        default="/data/data_tip_bag_pool_dino",
        help="Path to the output directory"
    )
    
    
    # Split arguments
    parser.add_argument("--train_split", type=int, default=80,
                        help="Training split percentage (int)")
    parser.add_argument("--val_split", type=int, default=20,
                        help="Validation split percentage (int)")

    # Ablation argument
    parser.add_argument("--ablation_size", type=int, default=0,
                        help="Ablation size (int)")
    

    args = parser.parse_args()
    
    
    # --- Validation ---
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: {args.input_dir} is not a valid directory")

    if not os.path.isdir(args.output_dir):
        sys.exit(f"Error: {args.output_dir} is not a valid directory")

    if args.train_split + args.val_split != 100:
        sys.exit("Error: train_split + val_split must equal 100")

    if args.train_split < 0 or args.val_split < 0 or args.ablation_size < 0:
        sys.exit("Error: all numeric arguments must be non-negative")

    return args


def main():    
    args = get_args()

    SRC_DIR = pathlib.Path(args.input_dir)
    DEST_DIR = pathlib.Path(args.output_dir)
    ABLATION_SIZE = args.ablation_size
    train_split, val_split = (args.train_split / 10, args.val_split / 10)

    print(f"Cleaning Destination directory: {DEST_DIR}")
    shutil.rmtree(DEST_DIR, ignore_errors=True)
    os.makedirs(DEST_DIR, exist_ok=True)

    # collect labels
    labels = os.listdir(SRC_DIR)
    for lbl in labels:
        # collect images
        dname = pathlib.Path(SRC_DIR) / lbl
        items = [item for item in os.listdir(dname) if item.endswith(".png") and os.path.isfile(dname / item)]
        
        # shuffle randomly
        random.seed(1)
        random.shuffle(items)        
       
        # create split sizes
        l = len(items)
        print("Image (label : count):",lbl,":", l)
        train_size = int(l * (train_split / 10))
        
        # create split list
        train_items = items[ : train_size]
        val_items = items[train_size : ]
        
        if ABLATION_SIZE > 0:
            train_items = items[:ABLATION_SIZE]
            val_items = items[:ABLATION_SIZE]
        
        
        # create train.txt & val.txt
        for collection_file, collection in zip(("train.txt", "val.txt"), (train_items, val_items)):
            with open(DEST_DIR / collection_file, "w") as fp:
                for item in collection:
                    fp.write("%s,%s\n" % (os.path.abspath(dname / item), lbl))
            print("Created: ", str(DEST_DIR / collection_file))

if __name__ == "__main__":
    main()
