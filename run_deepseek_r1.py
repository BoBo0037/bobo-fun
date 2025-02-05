# import os
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from utils.helper import set_device
from src.DeepSeekManager import DeepSeekManager

# - model list:
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# set device
device = set_device()
dtype = torch.bfloat16

# init & setup model
deepseek = DeepSeekManager(device, dtype, MODEL_ID)
deepseek.setup()

# input
input_text = "光速是多少?"

# infer
query = [
    {"role": "user", "content": input_text}
]
print(deepseek.infer(query, True))

# release
deepseek.cleanup()
