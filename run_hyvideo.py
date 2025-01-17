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

# 640 * 400: 46frames, ~3s
# 480 * 352: 61frames, ~4s
hyvideo.set_output_layout(
    width=480, 
    height=352,
    num_frames=61,
    num_inference_steps=30,
    fps=15
)

# setup
hyvideo.setup()

# generate video
hyvideo.generate()

# release
hyvideo.cleanup()
