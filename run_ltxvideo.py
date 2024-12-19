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

# refer settings can make a video on Mac: 
# [768, 512], fps=24, num_frames = 121, around 5 seconds video
# [1280, 720], fps=24, num_frames = 49, around 2 seconds video
ltx.set_output_layout(
    width=768, 
    height=512, 
    frame_rate=24, 
    num_frames=121, 
    num_inference_steps=50
)

ltx.set_stg(
    stg_mode="stg-a",   # Choose between 'stg-a' or 'stg-r'
    stg_scale=0,      # Recommended values are ≤2.0, (stg_scale = 0.0 means do not using stg)
    stg_block_idx=[],   # Specify the block index for applying STG
    do_rescaling=True  # Set to True to enable rescaling
)

#ltx.set_input_image("assets/imgs/suv.png")

# setup
ltx.setup()

# generate
ltx.generate()

# release
ltx.cleanup()
