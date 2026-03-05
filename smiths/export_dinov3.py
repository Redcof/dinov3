# Convert a torch-distributed-checkpoint to plain torch checkpoint
# This will generate a '<student/teacher>-torch.pth'
# IMPORTANT: Do not specify a *.distcp file, instead specify the root directory.
#
# Example: sh smiths/distcp2pth.py outputs/hif_dinov3/ckpt/999 /path/to/output/dir
#
# Author: Soumen Sardar
# Date: 2026-02-24

import os
import pathlib
import argparse
from dinov3.train.ssl_meta_arch import SSLMetaArch
import torch
from omegaconf import OmegaConf
from contextlib import suppress


def load_model(input_path):
    # load the config from the checkpoint directory
    cfg = read_config(input_path)
    # load the checkpoint using torch.distributed.checkpoint format_utils
    state_dict = load_checkpoint(input_path)
    # load the model architecture and weights
    model = SSLMetaArch(cfg)
    model.to_empty(device='cpu')
    model.load_state_dict(state_dict['model'], strict=True) # load checkpoint
    return model

def load_checkpoint(input_path):
    # load the checkpoint using torch.distributed.checkpoint format_utils
    pth_interm_file = distcp2pth(input_path)
    state_dict = torch.load(pth_interm_file, weights_only=False)
    os.remove(pth_interm_file) # remove the intermidiate file
    return state_dict

def distcp2pth(input_path):
    # create output files
    pth_interm_file = f"smiths-distcp2_intermidiate.pth"    
    with suppress(FileNotFoundError): os.remove(pth_interm_file) # remove the intermidiate file if it already exists
    os.system(r"python -m torch.distributed.checkpoint.format_utils dcp_to_torch %s %s" % (input_path, str(pth_interm_file)))
    return str(pth_interm_file)

def read_config(input_path):
    # load the config from the checkpoint directory
    cfg_file = pathlib.Path(input_path).parents[1] / "config.yaml"
    cfg = OmegaConf.load(cfg_file)
    return cfg

def remove_torch_hub_cache(pth_file):
    # remove the torch hub cache to avoid loading the old checkpoint
    cache_dir = torch.hub.get_dir()
    with suppress(FileNotFoundError):
        os.remove(str(pathlib.Path(cache_dir) / "checkpoints" / os.path.basename(pth_file)))

def verify_checkpoint(arch, pth_file):
    # verify the generated checkpoint by loading it and checking the keys
    remove_torch_hub_cache(pth_file)
    arch_name = dict(
        vit_small="dinov3_vits16",
        vit_base="dinov3_vitb16",
        vit_large="dinov3_vitl16",
        vit_7b="dinov3_vit7b16",
    )[arch]
    
    print(f"[Verifying...", end="")
    torch.hub.load(
        ".",
        arch_name,
        weights=str(pth_file),
        source="local",
    )
    print(f"DONE")

def get_git_sha():
    # get the git sha of the current commit
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception as e:
        print(f"Warning: Could not get git sha. {e}")
        sha = "unknown"
    return sha

def export_dinov3(input_path, output_dir):
    model = load_model(input_path)
    cfg = read_config(input_path)
    # read the model architecture and dataset information from the config to name the output files
    model_arch = cfg.student.arch
    dataset = cfg.train.dataset_path.split(":")[0]
    
    sha = get_git_sha()
    
    print(f"Saving the student and teacher backbone weights to separate torch checkpoint files...")
    pth_student_file = pathlib.Path(output_dir) / f"smiths-dinov3_{model_arch}_{dataset}_student_{sha}.pth"
    pth_teacher_file = pathlib.Path(output_dir) / f"smiths-dinov3_{model_arch}_{dataset}_teacher_{sha}.pth"
    
    with suppress(FileNotFoundError): os.remove(pth_student_file)
    with suppress(FileNotFoundError): os.remove(pth_teacher_file)
    torch.save(model.student.backbone.to('cpu').state_dict(), pth_student_file)
    torch.save(model.teacher.backbone.to('cpu').state_dict(), pth_teacher_file)
    
    # verify the generated checkpoints
    verify_checkpoint(model_arch, pth_student_file)

def main():
    parser = argparse.ArgumentParser(description="Python tool to convert dinov3 checkpoint to torch checkpoint")
    # 2. Add the first positional argument
    parser.add_argument("distributed_checkpoint_path", type=str, 
                        help="A dinov3 distributed checkpoint path (the root directory, not the .distcp file).")
    # 3. Add the second positional argument
    parser.add_argument("--output-dir", type=str,  default=".", 
                        help="The output directory where the torch checkpoint will be saved."
                        "The generated checkpoint will be named 'dinov3_torch.pth' and will be "
                        "located at <output-dir>/dinov3_torch.pth. Default=current directory.")
    # 4. Parse the arguments
    args = parser.parse_args()

    input_path = args.distributed_checkpoint_path
    output_dir = args.output_dir
    
    export_dinov3(input_path, output_dir)
    

if __name__ == "__main__":
    main()