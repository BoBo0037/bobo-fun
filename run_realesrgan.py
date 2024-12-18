import torch
from utils.helper import set_device
from src.RealESRGANManager import RealESRGANManager

# Init
device = set_device()
realesrgan_manager = RealESRGANManager(device, torch.bfloat16)
args = realesrgan_manager.get_args()

# handle img
args.input = "imgs/BlackCatDetective.jpg"
args.model_name = "realesr-general-x4v3"
args.outscale=1
args.face_enhance=True
args.output = "output_RealESRGAN"
realesrgan_manager.process_img(args)

# handle video
args.input = "videos/onepiece_demo.mp4"
args.model_name = "realesr-general-x4v3"
args.outscale=1
args.face_enhance=True
args.output = "output_RealESRGAN"
realesrgan_manager.process_video(args)
