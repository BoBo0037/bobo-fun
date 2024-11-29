import torch
from utils.helper import set_device
from src.PhiManager import PhiManager

# set device
device = set_device()
dtype = torch.bfloat16

# init & setup model
phi = PhiManager(device, dtype, "microsoft/Phi-3.5-mini-instruct")
phi.setup()

# infer
query = [
    {"role": "system", "content": "你是一个只说中文的AI小助手"},
    {"role": "user", "content": "光速是多少？"}
]
print(phi.infer(query), flush=True)

# release
phi.cleanup()
