from utils.helper import set_device
import torch
from src.HunyuanVideoManager import HunyuanVideoManager
from src.PromptManager import PromptManager

# Init
device = set_device()
hyvideo = HunyuanVideoManager(device, torch.bfloat16)

hyvideo.setup()

hyvideo.generate()

hyvideo.cleanup()
