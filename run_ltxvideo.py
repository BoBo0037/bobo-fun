import torch
from utils.helper import set_device
from src.LTXvideoManager import LTXVideoManager
from src.PromptManager import PromptManager

# set device
device = set_device()
ltx = LTXVideoManager(device, torch.bfloat16)

# set params
ltx.set_prompt(prompt=PromptManager("prompts.json").get("toy-ship"))

ltx.set_output_layout(
    width= 736, 
    height=480, 
    frame_rate=25, 
    num_frames=129, 
    num_inference_steps=40
)
#ltx.set_input_image("imgs/panda.png")

# setup
ltx.setup()

# generate
ltx.generate()

# release
ltx.cleanup()
