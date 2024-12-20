from utils.helper import set_device, check_numpy_version
from src.RIFEManager import RIFEManager
import torch
torch.set_default_tensor_type(torch.HalfTensor)

# check numpy
check_numpy_version()

# Init
device = set_device()
rife_manager = RIFEManager(device, torch.bfloat16)
args = rife_manager.get_args()

# setup model
rife_manager.setup(args)

# handle img
args.img = ["assets/imgs/boy/one.png", "assets/imgs/boy/two.png"]
args.output = "output_RIFE"
args.exp = 4
rife_manager.process_img(args)

# handle video
args.video = "assets/vids/dog.mp4"
args.output = "output_RIFE/interpolated_video.mp4"
args.exp = 2
args.scale = 1.0
args.fp16 = True
rife_manager.process_video(args)

# release
rife_manager.cleanup()
