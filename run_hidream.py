from utils.helper import set_device
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from src.HiDreamManager import HiDreamManager
from src.PromptManager import PromptManager

# init
device = set_device()
hidream_manager = HiDreamManager(device, torch.bfloat16)

# settings
hidream_manager.set_prompt(PromptManager("prompts.json").get("BMW"))
hidream_manager.set_output_layout(width=256, height=256)

# setup
hidream_manager.setup()

# generate images
hidream_manager.generate()

# release
hidream_manager.cleanup()
