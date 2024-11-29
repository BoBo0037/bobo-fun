import gc
import json
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional

import imageio
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from .ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from .ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from .ltx_video.models.transformers.transformer3d import Transformer3DModel
from .ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from .ltx_video.schedulers.rf import RectifiedFlowScheduler
from .ltx_video.utils.conditioning_method import ConditioningMethod

from huggingface_hub import snapshot_download

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

logger = logging.get_logger("LTXVideoManager")

class LTXVideoManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.model : str = "Lightricks/LTX-Video"
        self.model_sub: str = "PixArt-alpha/PixArt-XL-2-1024-MS"
        self.ckpt_dir : str = os.path.expanduser("~/.cache/huggingface/hub/models--Lightricks--LTX-Video/")
        self.prompt : str = "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."
        self.negative_prompt : str = "worst quality, inconsistent motion, blurry, jittery, distorted"
        self.output_path : str = None
        self.input_image_path : str = None
        self.input_video_path: str = None
        self.width : int = 704
        self.height : int = 480
        self.frame_rate: int = 24
        self.num_frames : int = 121
        self.guidance_scale: float = 3.0
        self.num_images_per_prompt : int = 1
        self.num_inference_steps: int = 50
        self.seed : int = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.cuda.empty_cache()

    def setup(self):
        # seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        seed_everething(self.seed)
        print(f"Using seed: {self.seed}")

        # output path
        self.output_dir = (
            Path(self.output_path)
            if self.output_path
            else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output direction: {self.output_dir}")

        # check model
        print("Check and download model")
        snapshot_download(repo_id=self.model, 
                          local_dir=self.ckpt_dir,
                          repo_type='model', 
                          ignore_patterns=["ltx-video-2b-v0.9.safetensors", "media/*"])
        text_encoder = T5EncoderModel.from_pretrained(self.model_sub, torch_dtype=self.dtype, subfolder="text_encoder").to(self.device)
        tokenizer = T5Tokenizer.from_pretrained(self.model_sub, torch_dtype=self.dtype, subfolder="tokenizer")
        print("Finish prepare model")

        # Load image
        if self.input_image_path:
            self.media_items_prepad = load_image_to_tensor_with_resize_and_crop(
                self.input_image_path, self.height, self.width
            )
        else:
            self.media_items_prepad = None

        height = self.height if self.height else self.media_items_prepad.shape[-2]
        width = self.width if self.width else self.media_items_prepad.shape[-1]
        num_frames = self.num_frames

        if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
            logger.warning(f"Input resolution or number of frames {height}x{width}x{num_frames} is too big, it is suggested to use the resolution below {MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES}.")

        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        self.height_padded = ((height - 1) // 32 + 1) * 32
        self.width_padded = ((width - 1) // 32 + 1) * 32
        self.num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
        logger.warning(f"Padded dimensions: {self.height_padded}x{self.width_padded}x{self.num_frames_padded}")

        self.padding = calculate_padding(height, width, self.height_padded, self.width_padded)

        if self.media_items_prepad is not None:
            self.media_items = F.pad(
                self.media_items_prepad, self.padding, mode="constant", value=-1
            )  # -1 is the value for padding since the image is normalized to -1, 1
        else:
            self.media_items = None

        # Prepare input for the pipeline
        self.sample = {
            "prompt": self.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": self.negative_prompt,
            "negative_prompt_attention_mask": None,
            "media_items": self.media_items,
        }

        # Paths for the separate mode directories
        ckpt_dir = Path(self.ckpt_dir)
        unet_dir = ckpt_dir / "unet"
        vae_dir = ckpt_dir / "vae"
        scheduler_dir = ckpt_dir / "scheduler"

        # Load models
        vae = load_vae(vae_dir)
        unet = load_unet(unet_dir).to(self.device)
        scheduler = load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)

        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": unet,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
        }
        self.pipeline = LTXVideoPipeline(**submodel_dict).to(self.device)
        self.generator=torch.Generator().manual_seed(self.seed)

    def generate(self):
        print("Start generate video")
        images = self.pipeline(
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=self.num_images_per_prompt,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            output_type="pt",
            callback_on_step_end=None,
            height=self.height_padded,
            width=self.width_padded,
            num_frames=self.num_frames_padded,
            frame_rate=self.frame_rate,
            **self.sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=(
                ConditioningMethod.FIRST_FRAME
                if self.media_items is not None
                else ConditioningMethod.UNCONDITIONAL
            ),
            mixed_precision=False,
        ).images

        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = self.padding
        pad_bottom = -pad_bottom
        pad_right = -pad_right
        if pad_bottom == 0:
            pad_bottom = images.shape[3]
        if pad_right == 0:
            pad_right = images.shape[4]
        images = images[:, :, :self.num_frames, pad_top:pad_bottom, pad_left:pad_right]

        for i in range(images.shape[0]):
            # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
            video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
            # Unnormalizing images to [0, 255] range
            video_np = (video_np * 255).astype(np.uint8)
            fps = self.frame_rate
            height, width = video_np.shape[1:3]
            # In case a single image is generated
            if video_np.shape[0] == 1:
                output_filename = get_unique_filename(
                    f"image_output_{i}",
                    ".png",
                    prompt=self.prompt,
                    seed=self.seed,
                    resolution=(height, width, self.num_frames),
                    dir=self.output_dir,
                )
                imageio.imwrite(output_filename, video_np[0])
            else:
                if self.input_image_path:
                    base_filename = f"img_to_vid_{i}"
                else:
                    base_filename = f"text_to_vid_{i}"
                output_filename = get_unique_filename(
                    base_filename,
                    ".mp4",
                    prompt=self.prompt,
                    seed=self.seed,
                    resolution=(height, width, self.num_frames),
                    dir=self.output_dir,
                )

                # Write video
                with imageio.get_writer(output_filename, fps=fps) as video:
                    for frame in video_np:
                        video.append_data(frame)

                # Write condition image
                if self.input_image_path:
                    reference_image = (
                        (
                            self.media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy()
                            + 1.0
                        )
                        / 2.0
                        * 255
                    )
                    imageio.imwrite(
                        get_unique_filename(
                            base_filename,
                            ".png",
                            prompt=self.prompt,
                            seed=self.seed,
                            resolution=(height, width, self.num_frames),
                            dir=self.output_dir,
                            endswith="_condition",
                        ),
                        reference_image.astype(np.uint8),
                    )
            logger.warning(f"Output saved to {self.output_dir}")

    def set_prompt(self, 
                   prompt : str,
                   negative_prompt : Optional[str] = "worst quality, inconsistent motion, blurry, jittery, distorted") -> None:
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        print(f"Set prompt to '{self.prompt}'")
        print(f"Set negative prompt to '{self.negative_prompt}'")

    def set_input_image(self, input_image_path : str) -> None:
        self.input_image_path = input_image_path
        print(f"Set input image to '{self.input_image_path}'")

    def set_input_video(self, input_video_path : str) -> None:
        self.input_video_path = input_video_path
        print(f"Set input video to '{self.input_video_path}'")
    
    def set_output_layout(self, 
                          output_path : Optional[str] = None, 
                          width : Optional[int] = 704, 
                          height : Optional[int] = 480, 
                          frame_rate : Optional[int] = 24, 
                          num_frames : Optional[int] = 121,
                          num_inference_steps : Optional[int] = 50) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        print(f"Set output path to '{self.output_path}'")
        print(f"Set video width and height to '{self.width}, {self.height}'")
        print(f"Set video frame rate and num of frames to '{self.frame_rate}' and '{self.num_frames}'")
        print(f"Set num of inference steps to '{self.num_inference_steps}'")


def load_vae(vae_dir):
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    if torch.cuda.is_available():
        vae = vae.cuda()
    return vae.to(torch.bfloat16)


def load_unet(unet_dir):
    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
    unet_config_path = unet_dir / "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    return transformer


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)


def load_image_to_tensor_with_resize_and_crop(
    image_path, target_height=512, target_width=768
):
    image = Image.open(image_path).convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)

# Generate output video name
def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )


def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
