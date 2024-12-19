import os
import subprocess
import argparse
import logging
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import torch.jit
from typing import Optional
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
from src.mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()
from src.mimicmotion.constants import ASPECT_RATIO
from src.mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from src.mimicmotion.utils.loader import create_pipeline
from src.mimicmotion.utils.utils import save_to_mp4
from src.mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CKPT_ID = "tencent/MimicMotion"
CKPT_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--tencent--MimicMotion")
CKPT_MODEL_NAME = CKPT_PATH + "/MimicMotion_1.pth"

MODEL_DWPOSE_ID = "yzd-v/DWPose"
MODEL_DWPOSE_DIR = CKPT_PATH + "/DWPose"

BASE_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
BASE_MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/")

class MimicManager:
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.ref_video_path = "./assets/vids/dance_demo.mp4"
        self.ref_image_path = "./assets/imgs/1girl.jpg"
        self.resolution = 576
        self.fps = 15
        self.num_frames = 72
        self.num_inference_steps = 25
        self.frames_overlap = 6
        self.sample_stride = 2
        self.guidance_scale = 2.0
        self.noise_aug_strength = 0
        self.seed = None

    def download_model(self) -> None: 
        # models/
        # ├── DWPose
        # │   ├── dw-ll_ucoco_384.onnx
        # │   └── yolox_l.onnx
        # └── MimicMotion_1.pth
        print("Start download models")
        command_download_tencent_mimic_motion = [
            "huggingface-cli",
            "download",
            CKPT_ID,
            "--local-dir",
            CKPT_PATH,
            "--include",
            "MimicMotion_1.pth"
        ]
        subprocess.run(command_download_tencent_mimic_motion, env=os.environ)

        command_download_dwpose = [
            "huggingface-cli",
            "download",
            MODEL_DWPOSE_ID,
            "--local-dir",
            MODEL_DWPOSE_DIR,
            "--include",
            "yolox_l.onnx",
            "dw-ll_ucoco_384.onnx"
        ]
        subprocess.run(command_download_dwpose, env=os.environ)
        
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path = BASE_MODEL_ID,
            torch_dtype = self.dtype,
            variant="fp16",
            use_safetensors = True
        )
        pipe.save_pretrained(BASE_MODEL_DIR,  variant="fp16", use_safetensors = True)
        print("Finish prepare models")
    
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--log_file", type=str, default=None)
        parser.add_argument("--inference_config", type=str, default="src/mimic_configs.yaml")
        parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
        parser.add_argument("--no_use_float16", action="store_true", help="Whether use float16 to speed up inference")
        args = parser.parse_args()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        set_logger(args.log_file if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
        self.process(args)
        logger.info(f"--- Finished ---")

    def process(self, args):
        if not args.no_use_float16 :
            torch.set_default_dtype(torch.float16)

        #infer_config = OmegaConf.load(args.inference_config)
        pipeline = create_pipeline(BASE_MODEL_DIR, CKPT_MODEL_NAME, self.device)

        ############################################## Pre-process data ##############################################
        print("Start preprocess")
        pose_pixels, image_pixels = preprocess(
            self.ref_video_path, 
            self.ref_image_path, 
            resolution=self.resolution, 
            sample_stride=self.sample_stride
        )
        ########################################### Run MimicMotion pipeline ###########################################
        print("Start generate video")
        _video_frames = self.run_pipeline(
            pipeline, 
            image_pixels, 
            pose_pixels
        )
        ################################### save results to output folder. ###########################################
        print("Start save video")
        save_to_mp4(
            _video_frames, 
            f"{args.output_dir}/{os.path.basename(self.ref_video_path).split('.')[0]}" \
            f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
            fps=self.fps
        )

    @torch.inference_mode()
    def run_pipeline(self, pipeline: MimicMotionPipeline, image_pixels, pose_pixels):
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        frames = pipeline(
            image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=self.num_frames, tile_overlap=self.frames_overlap,
            height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
            noise_aug_strength=self.noise_aug_strength, num_inference_steps=self.num_inference_steps,
            generator=generator, min_guidance_scale=self.guidance_scale, 
            max_guidance_scale=self.guidance_scale, decode_chunk_size=8, output_type="pt", device=self.device
        ).frames.cpu()
        video_frames = (frames * 255.0).to(torch.uint8)
        for vid_idx in range(video_frames.shape[0]):
            # deprecated first frame because of ref image
            _video_frames = video_frames[vid_idx, 1:]
        return _video_frames

    def set_output_layout(self, 
                          ref_video_path : str = None, 
                          ref_image_path : str = None, 
                          resolution : Optional[int] = 576,
                          fps : Optional[int] = 15, 
                          num_frames : Optional[int] = 72,
                          num_inference_steps : Optional[int] = 25,
                          frames_overlap : Optional[int] = 6,
                          sample_stride : Optional[int] = 2,
                          guidance_scale : Optional[float] = 2.0,
                          ) -> None:
        self.ref_video_path = ref_video_path
        self.ref_image_path = ref_image_path
        self.resolution = resolution
        self.fps = fps
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.frames_overlap = frames_overlap
        self.sample_stride = sample_stride
        self.guidance_scale = guidance_scale
        print(f"Set ref video path to '{self.ref_video_path}'")
        print(f"Set ref image path to '{self.ref_image_path}'")
        print(f"Set resolution to '{self.resolution}'")
        print(f"Set fps to '{self.fps}'")
        print(f"Set num of frames to '{self.num_frames}'")
        print(f"Set num of inference steps to '{self.num_inference_steps}'")
        print(f"Set frames overlap to '{self.frames_overlap}'")
        print(f"Set sample_stride to '{self.sample_stride}'")
        print(f"Set guidance_scale to '{self.guidance_scale}'")

def preprocess(video_path, image_path, resolution=576, sample_stride=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    ##################################### get image&video pose value #################################################
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)
