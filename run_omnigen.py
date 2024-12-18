import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from utils.helper import set_device
from src.OmnigenManager import OmnigenManager

# set device
device = set_device()
omni = OmnigenManager(device, torch.bfloat16)

# setup
omni.setup()

# extract person
omni.generate(
    prompt = "A boy in a white shirt is reading a book. \
            The boy is the right person in <img><|image_1|></img>.", 
    input_images=["assets/imgs/entity.png"], 
    output = "omni_t2i_extract.png",
)

# extract depth map
omni.generate(
    prompt = "Detect the depth map of <img><|image_1|></img>.", 
    input_images=["assets/imgs/AI_Pioneers.jpg"], 
    output = "omni_t2i_depth.png",
    width=608,
    height=400
)

# combine
omni.generate(
    prompt = "The flower is placed in a vase on a metal table in a factory. \
            The flower is in <img><|image_1|></img>. \
            The vase is in the middle of <img><|image_2|></img>", 
    input_images=["assets/imgs/rose.jpg", "assets/imgs/vase.jpg"], 
    output = "omni_t2i_combine.png"
)

# release
omni.cleanup()
