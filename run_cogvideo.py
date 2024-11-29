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
    model_path="THUDM/CogVideoX-2b", # "THUDM/CogVideoX-2b", "NimVideo/cogvideox-2b-img2vid" "THUDM/CogVideoX1.5-5B", "THUDM/CogVideoX1.5-5B-I2V" 
    generate_type="t2v"              # Literal["t2v", "i2v", "v2v"]
)

# only can generate 2 seconds on mac ?
cogVideo.set_output_layout(
    output_path="./output.mp4", 
    width= 720, 
    height=480,
    fps=8,
    num_frames=17,
    num_inference_steps=30
)

#cogVideo.set_input_image_or_video("imgs/panda.png")

# generate
cogVideo.generate()

# release
cogVideo.cleanup()
