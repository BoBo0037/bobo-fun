import torch
from utils.helper import set_device
from src.CogVideoFunManager import CogVideoFunManager
from src.PromptManager import PromptManager

# Init
should_download = False # should download at first time only
device = set_device()
cogVideoFun = CogVideoFunManager(device, torch.bfloat16)

# rewrite prompt to video mode
prompt_manager = PromptManager("prompts.json")
original_prompt = prompt_manager.get("suv")
#rewrited_prompt = prompt_manager.rewriter.rewrite_video_prompt(original_prompt)

# refer settings can make a video on Mac: 
# [768, 432], fps=8, num_frames = 24, around 3 seconds video
# [640, ???], fps=8, num_frames = 33, around 4 seconds video
# [480, 272], fps=8, num_frames = 41, around 5 seconds video
cogVideoFun.set_prompt(prompt=original_prompt)
cogVideoFun.set_validation_images(validation_image_start="assets/imgs/suv.png")
cogVideoFun.set_output_layout(
    output_path="output_cogvideox_fun", 
    width= 480, 
    height=272,
    fps=8,
    num_frames=41,
    num_inference_steps=50
)

# download model if needed
if should_download:
    print("start download models")
    cogVideoFun.download()

# setup
cogVideoFun.setup()

# t2v case:
cogVideoFun.run_t2v()

# i2v case:
#cogVideoFun.run_i2v()

# release
cogVideoFun.cleanup()
