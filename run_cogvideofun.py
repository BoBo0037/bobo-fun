import torch
from utils.helper import set_device
from src.CogVideoFunManager import CogVideoFunManager
from src.PromptManager import PromptManager

# Init
should_download = True # should download at first time only
device = set_device()
cogVideoFun = CogVideoFunManager(device, torch.bfloat16)

# settings
cogVideoFun.set_prompt(prompt=PromptManager("prompts.json").get("suv"))
cogVideoFun.set_validation_images(validation_image_start="imgs/suv.png")
cogVideoFun.set_output_layout(
    output_path="output_cogvideox_fun", 
    width= 768, 
    height=432,
    fps=8,
    num_frames=24,
    num_inference_steps=50
)

# download model if needed
if should_download:
    print("start download models")
    cogVideoFun.download()

# setup
cogVideoFun.setup()

# t2v case:
#cogVideoFun.run_t2v()

# i2v case:
cogVideoFun.run_i2v()

# release
cogVideoFun.cleanup()
