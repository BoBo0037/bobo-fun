
from utils.helper import set_device
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from src.WanVideoManager import WanVideoManager, replace_wan_transformer, revert_wan_transformer
from src.PromptManager import PromptManager

# temporarily replace WanRotaryPosEmbed and WanAttnProcessor2_0.call for mps device
original_class, original_call = replace_wan_transformer()

# init
device = set_device()
wanvideo = WanVideoManager(device, torch.bfloat16)

# settings
wanvideo.set_prompt(PromptManager("prompts.json").get("suv"))

wanvideo.set_output_layout(
    width=832, 
    height=480,
    num_frames=9,
    num_inference_steps=30,
    fps=8
)

# setup
wanvideo.setup()

# generate video
wanvideo.generate()

# release
wanvideo.cleanup()

# revert
revert_wan_transformer(original_class, original_call)
