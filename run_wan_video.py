
from utils.helper import set_device
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from src.WanVideoManager import WanVideoManager, replace_wan_transformer, revert_wan_transformer
from src.PromptManager import PromptManager

# temporarily replace WanRotaryPosEmbed and WanAttnProcessor2_0.call for mps device
original_class, original_call = replace_wan_transformer()

# set whether or not using i2v
enable_i2v = True

# init
device = set_device()
wanvideo = WanVideoManager(device, torch.bfloat16)

# settings
if enable_i2v:
    wanvideo.set_prompt("A young girl strolls in the evening.")
    wanvideo.set_image("assets/imgs/animation_girl.png")
else:
    wanvideo.set_prompt(PromptManager("prompts.json").get("suv"))

wanvideo.set_output_layout(
    width=432, 
    height=768,
    num_frames=17,
    num_inference_steps=10,
    fps=8
)

# setup
wanvideo.setup(enable_i2v=enable_i2v)

# generate video
wanvideo.generate(enable_i2v=enable_i2v)

# release
wanvideo.cleanup()

# revert
revert_wan_transformer(original_class, original_call)
