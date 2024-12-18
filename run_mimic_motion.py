from utils.helper import set_device
from src.MimicManager import MimicManager
import torch

# Init
need_download = False
device = set_device()
mimic_manager = MimicManager(device, torch.float16)
# prepare models
if need_download:
    mimic_manager.download_model()
# set args
mimic_manager.set_output_layout(
    ref_video_path="./assets/vids/dance_demo.mp4", 
    ref_image_path="./assets/imgs/1girl.jpg",
    fps=8,
    num_frames=17,
    num_inference_steps=20,
    frames_overlap=4
)
# generate video
mimic_manager.run()
