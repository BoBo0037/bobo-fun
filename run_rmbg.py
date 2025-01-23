from utils.helper import set_device
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from src.RMBGManager import RMBGManager

device = set_device()
rmbg = RMBGManager(device, torch.bfloat16)

input_image = "assets/imgs/AI_Pioneers.jpg"
background_image = "assets/imgs/xp.jpg"

# setup
rmbg.setup()

# generate
rmbg.generate(input_image, background_image)

# release
rmbg.cleanup()
