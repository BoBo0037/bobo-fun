import torch
from utils.helper import set_device
from src.MochiManager import MochiManager
from src.PromptManager import PromptManager

# set device
device = set_device()
mochi = MochiManager(device, torch.bfloat16)

# set params
mochi.set_prompt(
    prompt=PromptManager("prompts.json").get("suv"),
    negative_prompt=PromptManager("prompts.json").get("negative-video")
)

mochi.set_output_layout(
    width=432,  # [416, 848] 
    height=256, # [240, 480]
    fps=6,      # just using fps in the save video process
    num_frames=7, 
    num_inference_steps=64 # have to 64
)

# setup
mochi.setup()

# generate
mochi.generate()

# release
mochi.cleanup()
