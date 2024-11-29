import torch
from utils.helper import set_device
from src.GlmManager import GlmManager

# set device
device = set_device()
dtype = torch.bfloat16

# init & setup model
glm = GlmManager(device, dtype, "THUDM/glm-4-9b-chat")
glm.setup()

# infer
query = [
    {"role": "system", "content": "你是一个只说中文的AI小助手"},
    {"role": "user", "content": "光速是多少？"}
]
print(glm.infer(query))

# release
glm.cleanup()
