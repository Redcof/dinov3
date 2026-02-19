# This script create train.txt and val.txt file in a destination folder
# with (absolute-file-path,label) data per line
# Author: Soumen Sardar
# Date: 2026-02-18
# ####################################################################### 

import os
import pathlib
import shutil
import random


SRC_DIR = pathlib.Path("/data/data_tip_bag_pool_png")
DEST_DIR = pathlib.Path("/data/data_tip_bag_pool_dino")
ABLATION_SIZE = 20
train_split, val_split = (8, 2)

shutil.rmtree(DEST_DIR, ignore_errors=True)
os.makedirs(DEST_DIR, exist_ok=True)

# def create_link(src, dest):
#     # Create the symbolic link
#     try:
#         os.co(os.path.abspath(src), dest / os.path.basename(src))
#     except FileExistsError:
#         pass

def main():
    # collect labels
    labels = os.listdir(SRC_DIR)
    for lbl in labels:
        # collect images
        dname = pathlib.Path(SRC_DIR) / lbl
        items = [item for item in os.listdir(dname) if item.endswith(".png") and os.path.isfile(dname / item)]
        
        # shuffle randomly
        random.seed(1)
        random.shuffle(items)
        
        # # create split-dirs
        # train_dir = pathlib.Path(DEST_DIR) / "train"/ lbl
        # val_dir = pathlib.Path(DEST_DIR) / "val"/ lbl
        # os.makedirs(train_dir, exist_ok=True)
        # os.makedirs(val_dir, exist_ok=True)
        
        # create split sizes
        l = len(items)
        print("Image (label : count):",lbl,":", l)
        train_size = int(l * (train_split / 10))
        
        # create split list
        train_items = items[ : train_size][:ABLATION_SIZE]
        val_items = items[train_size : ][:ABLATION_SIZE]
        
        if ABLATION_SIZE > 1:
            train_items = items[:ABLATION_SIZE]
            val_items = items[:ABLATION_SIZE]
        
        # # create links for split items
        # for collection, dest in zip((train_items, val_items), (train_dir, val_dir)):
        #     for file in collection:
        #         create_link(file, dest)
        
        # create train.txt & val.txt
        for collection_file, collection in zip(("train.txt", "val.txt"), (train_items, val_items)):
            with open(DEST_DIR / collection_file, "w") as fp:
                for item in collection:
                    fp.write("%s,%s\n" % (os.path.abspath(dname / item), lbl))
            print("Created: ", str(DEST_DIR / collection_file))

if __name__ == "__main__":
    main()