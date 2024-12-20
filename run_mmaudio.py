from utils.helper import set_device
from src.MMAudioManager import MMAudioManager
import torch

# Init
need_download = False
device = set_device()
mimic_manager = MMAudioManager(device, torch.float32)

# set args
mimic_manager.set_output_layout(
    prompt = "An suv is driving quickly on a mountain road.", 
    negative_prompt = "music",
    video = "assets/vids/suv1.mp4",
    duration = 4.0,
    num_steps = 25
)

# generate audio
mimic_manager.generate()

# release
mimic_manager.cleanup()
