import torch
from utils.helper import set_device
from src.LTXvideoManager import LTXVideoManager
from src.PromptManager import PromptManager

# set device
device = set_device()
ltx = LTXVideoManager(device, torch.bfloat16)

# set params
ltx.set_prompt(
    prompt=PromptManager("prompts.json").get("suv"),
    negative_prompt=PromptManager("prompts.json").get("negative-video")
)

ltx.set_output_layout(
    width= 768, 
    height=432, 
    frame_rate=8, 
    num_frames=24, 
    num_inference_steps=50
)
#ltx.set_input_image("imgs/panda.png")

# setup
ltx.setup()

# generate
ltx.generate()

# release
ltx.cleanup()
