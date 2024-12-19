import torch
from utils.helper import set_device
from src.CogVideoFunManager import CogVideoFunManager
from src.PromptManager import PromptManager

# Init
should_download = True # should download at first time only
device = set_device()
cogVideoFun = CogVideoFunManager(device, torch.bfloat16)

# rewrite prompt to video mode
prompt_manager = PromptManager("prompts.json")
original_prompt = prompt_manager.get("suv")
rewrited_prompt = prompt_manager.rewriter.rewrite_video_prompt(original_prompt)

# settings
# can make a video with on Mac: 
# fps=8, num_frames = 24, around 3 seconds video
# fps=6, num_frames = 24, around 4 seconds video
cogVideoFun.set_prompt(prompt=rewrited_prompt)
cogVideoFun.set_validation_images(validation_image_start="assets/imgs/suv.png")
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
cogVideoFun.run_t2v()

# i2v case:
#cogVideoFun.run_i2v()

# release
cogVideoFun.cleanup()
