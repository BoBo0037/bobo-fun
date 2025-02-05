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

# 1280 * 720: 20frames, 15fps, ~ 1.3s (Weird video BUG)
# 960 * 544 : 31frames, 15fps, ~ 2s (Weird video BUG)
# 640 * 400 : 46frames, 15fps, ~ 3s (ok)
# 480 * 352 : 61frames, 15fps, ~ 4s (ok)
hyvideo.set_output_layout(
    width=480, 
    height=352,
    num_frames=61,
    num_inference_steps=30,
    fps=15
)

# setup
hyvideo.setup_low_mem()
#hyvideo.setup()

# generate video
hyvideo.generate_low_mem()
#hyvideo.generate()

# release
hyvideo.cleanup()
