from utils.helper import set_device
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from src.HunyuanVideoManager import HunyuanVideoManager
from src.PromptManager import PromptManager

# Init
device = set_device()
hyvideo = HunyuanVideoManager(device, torch.bfloat16)

# settings
hyvideo.set_prompt(PromptManager("prompts.json").get("F-150"))
hyvideo.set_output_layout(
    width=720, 
    height=384,
    num_frames=65,
    num_inference_steps=35,
    fps=16
)

# setup
hyvideo.setup()

# generate video
hyvideo.generate()

# release
hyvideo.cleanup()
