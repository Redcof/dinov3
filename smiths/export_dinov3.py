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
import functools

PADDING = 0

def echo(message):
    """Outer function: takes the parameter n and returns the actual decorator."""
    def decorator(func):
        """Middle function: takes the function to be decorated."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Inner function: contains the main logic, uses the parameter n."""
            global PADDING
            if PADDING > 0:
                str_tab = "\t" * PADDING
            else:
                str_tab = ""
            print(str_tab, message)
            PADDING += 1
            result = func(*args, **kwargs)
            PADDING -= 1
            print(str_tab, "DONE")            
            return result
        # Returns the wrapper function, which replaces the original function
        return wrapper
    # Returns the decorator function
    return decorator

@echo("Loading the model... ")
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

@echo("Loading checkpoint... ")
def load_checkpoint(input_path):
    # load the checkpoint using torch.distributed.checkpoint format_utils
    pth_interm_file = distcp2pth(input_path)
    state_dict = torch.load(pth_interm_file, weights_only=False)
    os.remove(pth_interm_file) # remove the intermidiate file
    return state_dict

@echo("Converting distributed checkpoint to torch checkpoint... ")
def distcp2pth(input_path):
    # create output files
    pth_interm_file = f"smiths-distcp2_intermidiate.pth"    
    with suppress(FileNotFoundError): os.remove(pth_interm_file) # remove the intermidiate file if it already exists
    os.system(r"python -m torch.distributed.checkpoint.format_utils dcp_to_torch %s %s" % (input_path, str(pth_interm_file)))
    return str(pth_interm_file)

@echo("Reading config... ")
def read_config(input_path):
    # load the config from the checkpoint directory
    cfg_file = pathlib.Path(input_path).parents[1] / "config.yaml"
    cfg = OmegaConf.load(cfg_file)
    return cfg

@echo("Removing torch hub cache... ")
def remove_torch_hub_cache(pth_file):
    # remove the torch hub cache to avoid loading the old checkpoint
    cache_dir = torch.hub.get_dir()
    with suppress(FileNotFoundError):
        os.remove(str(pathlib.Path(cache_dir) / "checkpoints" / os.path.basename(pth_file)))

@echo("Verifying the generated checkpoint... ")
def verify_checkpoint(arch, pth_file):
    # verify the generated checkpoint by loading it and checking the keys
    remove_torch_hub_cache(pth_file)
    arch_name = dict(
        vit_small="dinov3_vits16",
        vit_base="dinov3_vitb16",
        vit_large="dinov3_vitl16",
        vit_7b="dinov3_vit7b16",
    )[arch]
    
    torch.hub.load(
        ".",
        arch_name,
        weights=str(pth_file),
        source="local",
    )
    
@echo("Calculating git sha... ")
def get_git_sha():
    # get the git sha of the current commit
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception as e:
        sha = "unknown"
    return sha

@echo("Exporting DINOv3 model... ")
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