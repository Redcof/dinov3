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
    
    pth_interm_file = pathlib.Path(output_dir) / "dinov3_privatedata_intermidiate.pth"
    pth_student_file = pathlib.Path(output_dir) / "dinov3_privatedata_student.pth"
    pth_teacher_file = pathlib.Path(output_dir) / "dinov3_privatedata_teacher.pth"
    
    print("Removing the intermediate torch checkpoint...")
    with suppress(FileNotFoundError): os.remove(pth_interm_file)
    with suppress(FileNotFoundError): os.remove(pth_student_file)
    with suppress(FileNotFoundError): os.remove(pth_teacher_file)

    print(f"Converting distributed to intermidiate torch checkpoint form '{input_path}'")
    os.system(r"python -m torch.distributed.checkpoint.format_utils dcp_to_torch %s %s" % (input_path, str(pth_interm_file)))
    
    
    print("Loading intermidiate torch checkpoint...")
    state_dict = torch.load(pth_interm_file, weights_only=False)
    
    cfg_file = pathlib.Path(input_path).parents[1] / "config.yaml"
    cfg = OmegaConf.load(cfg_file)
    model = SSLMetaArch(cfg)
    model.load_state_dict(state_dict['model'], strict=False) # load checkpoint
    
    print("Saving the student and teacher backbone weights to separate torch checkpoint files...")
    torch.save(model.student.backbone.state_dict(), pth_student_file)
    torch.save(model.teacher.backbone.state_dict(), pth_teacher_file)
    
    print("Removing the intermediate torch checkpoint...")
    with suppress(FileNotFoundError): os.remove(pth_interm_file)
    
    print(f"Generated torch checkpoint at: '{pth_student_file}'")
    print(f"Generated torch checkpoint at: '{pth_teacher_file}'")

if __name__ == "__main__":
    main()