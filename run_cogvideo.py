import torch
from utils.helper import set_device
from src.CogVideoManager import CogVideoManager
from src.PromptManager import PromptManager

# set device
device = set_device()
cogVideo = CogVideoManager(device, torch.bfloat16)

# set params
cogVideo.set_prompt(prompt=PromptManager("prompts.json").get("suv"))

cogVideo.set_model(
    model_path="THUDM/CogVideoX-2b", 
    generate_type="t2v"              # Literal["t2v", "i2v", "v2v"]
)

# can make a video with on Mac: 
# fps=8, num_frames = 24, around 3 seconds video
# fps=6, num_frames = 24, around 4 seconds video
cogVideo.set_output_layout(
    output_path="output_cogvideox", 
    width= 768, 
    height=432,
    fps=8,
    num_frames=24,
    num_inference_steps=50
)

#cogVideo.set_input_image_or_video("imgs/panda.png")

# generate
cogVideo.generate()

# release
cogVideo.cleanup()
