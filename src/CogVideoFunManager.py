

import gc
import os
import numpy as np
import torch
from PIL import Image
from typing import Literal, Optional
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from safetensors.torch import load_file, safe_open
from transformers import T5EncoderModel, T5Tokenizer
from src.cogvideoxfun.models.transformer3d import CogVideoXTransformer3DModel
from src.cogvideoxfun.models.autoencoder_magvit import AutoencoderKLCogVideoX
from src.cogvideoxfun.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from src.cogvideoxfun.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from src.cogvideoxfun.utils.lora_utils import merge_lora, unmerge_lora
from src.cogvideoxfun.utils.utils import get_image_to_video_latent, save_videos_grid
from utils.helper import check_and_make_folder

class CogVideoFunManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.model = "alibaba-pai/CogVideoX-Fun-V1.1-2b-InP"
        self.model_cache = os.path.expanduser("~/.cache/huggingface/hub/models--alibaba-pai--CogVideoX-Fun-V1.1-2b-InP")
        self.output_path = "output_cogvideox_fun"
        self.prompt = "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it's tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."
        self.negative_prompt = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."
        self.scheduler_dict = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler, 
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler
        }
        self.sampler_name = "DPM++"
        self.video_size = [432, 768]
        self.num_frames = 72
        self.fps = 8
        self.num_inference_steps = 50
        self.guidance_scale = 6.5
        self.seed = None # int.from_bytes(os.urandom(2), "big")
        self.low_gpu_memory_mode = False # have to 'False' on mac
        # lora
        self.lora_path        = None
        self.lora_weight      = 0.55
        # Set pretrained model if need
        self.transformer_path = None
        self.vae_path         = None
        # Image to video settings
        self.validation_image_start  = None
        self.validation_image_end    = None
        # Init
        self.pipe        = None
        self.transformer = None
        self.vae         = None
        self.scheduler   = None
        self.generator   = None
        self.samples     = None

    def cleanup(self):
        print("Run cleanup")
        torch.cuda.empty_cache()
        gc.collect()
    
    def setup(self):
        self.set_scheduler(self.sampler_name)
        self.load_transformer()
        self.load_vae()
        self.load_text_encoder()
        self.load_pipe()

    @torch.inference_mode()
    def run_t2v(self):
        print("start text to video")
        self.validation_image_start = None
        self.validation_image_end = None
        self.generate()
    
    @torch.inference_mode()
    def run_i2v(self):
        print("start image to video")
        self.generate()

    def generate(self):
        # merge lora
        self.merge_lora()
        # rescale num of frames
        self.rescale_num_frames()
        # prepare params
        params = self.set_params()
        # generate video
        self.samples = self.pipe(**params).videos
        # unmerge lora
        self.unmerge_lora()
        # save video
        self.save_video(self.output_path)
        print("finish generate video!")

    def save_video(self, output_path : str):
        print("start save video")

        # check output path
        check_and_make_folder(output_path)

        # get prefix
        index = len([path for path in os.listdir(output_path)]) + 1
        prefix = str(index).zfill(8)

        # save
        if self.samples is not None:
            if self.num_frames == 1:
                image_path = os.path.join(output_path, prefix + ".png")
                image = self.samples[0, :, 0]
                image = image.transpose(0, 1).transpose(1, 2)
                image = (image * 255).numpy().astype(np.uint8)
                image = Image.fromarray(image)
                image.save(image_path)
            else:
                video_path = os.path.join(output_path, prefix + ".mp4")
                save_videos_grid(self.samples, video_path, fps=self.fps)
        else:
            raise "cannot save video, video is None"

    def download(self):
        from diffusers import DiffusionPipeline
        #os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        pipe = DiffusionPipeline.from_pretrained(self.model, torch_dtype=self.dtype)
        pipe.save_pretrained(self.model_cache) # After save_pretrained(), you can remove duplicated weight files

    def load_transformer(self):
        print("prepare 'transformer'")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
            self.model_cache, 
            subfolder="transformer",
        ).to(self.dtype)

        if self.transformer_path is not None:
            print(f"From checkpoint: {self.transformer_path}")
            if self.transformer_path.endswith("safetensors"):
                state_dict = load_file(self.transformer_path)
            else:
                state_dict = torch.load(self.transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = self.transformer.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    def load_vae(self):
        print("prepare 'vae'")
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            self.model_cache, 
            subfolder="vae"
        ).to(self.dtype)

        if self.vae_path is not None:
            print(f"From checkpoint: {self.vae_path}")
            if self.vae_path.endswith("safetensors"):

                state_dict = load_file(self.vae_path)
            else:
                state_dict = torch.load(self.vae_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = self.vae.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    def load_text_encoder(self):
        print("prepare 'text_encoder'")
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.model_cache, 
            subfolder="text_encoder", 
            torch_dtype=self.dtype
        )

    def load_pipe(self):
        print("prepare 'pipeline'")
        if self.transformer.config.in_channels != self.vae.config.latent_channels:
            print("set pipe_name to 'CogVideoX_Fun_Pipeline_Inpaint'")
            pipe_name = CogVideoX_Fun_Pipeline_Inpaint
        else:
            print("set pipe_name to 'CogVideoX_Fun_Pipeline'")
            pipe_name = CogVideoX_Fun_Pipeline

        # load pipeline
        self.pipe = pipe_name.from_pretrained(
            self.model_cache,
            vae=self.vae,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            scheduler=self.scheduler,
            torch_dtype=self.dtype
        )
        # set to device
        self.pipe = self.pipe.to(self.device)

        # set seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        print(f"set seed to '{self.seed}'")

        # set generator
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        # set low gpu momory mode
        if not torch.backends.mps.is_available():
            if self.low_gpu_memory_mode:
                self.pipe.enable_sequential_cpu_offload()
            else:
                self.pipe.enable_model_cpu_offload()

    def set_params(self):
        # init params
        params = {
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'height': self.video_size[0],
            'width': self.video_size[1],
            'num_frames': self.num_frames,
            'guidance_scale': self.guidance_scale,
            'num_inference_steps': self.num_inference_steps,
            'generator': self.generator,
        }
        # update params
        if self.transformer.config.in_channels != self.vae.config.latent_channels:
            print("transformer.config.in_channels =", self.transformer.config.in_channels)
            print("vae.config.latent_channels =", self.vae.config.latent_channels)
            input_video, input_video_mask, clip_image = get_image_to_video_latent(
                validation_image_start = None if self.validation_image_start is None else self.validation_image_start, 
                validation_image_end = None if self.validation_image_end is None else self.validation_image_end, 
                video_length=self.num_frames, 
                sample_size=self.video_size
            )
            params.update({ 
                'video': input_video, 
                'mask_video': input_video_mask
            })
        return params

    def set_scheduler(self, sampler_name):
        print("set 'scheduler'")
        choosen_scheduler = self.scheduler_dict[sampler_name]
        self.scheduler = choosen_scheduler.from_pretrained(
            self.model_cache, 
            subfolder="scheduler"
        )

    def rescale_num_frames(self):
        compression_ratio = self.vae.config.temporal_compression_ratio
        self.num_frames = int((self.num_frames - 1) // compression_ratio * compression_ratio) + 1 if self.num_frames != 1 else 1
        print("num_frames reset to:", self.num_frames)

    def merge_lora(self):
        if self.lora_path is not None:
            print("merge lora")
            self.pipe = merge_lora(self.pipe, self.lora_path, self.lora_weight)

    def unmerge_lora(self):
        if self.lora_path is not None:
            print("unmerge lora")
            self.pipe = unmerge_lora(self.pipe, self.lora_path, self.lora_weight)

    def set_prompt(self, prompt : str) -> None:
        self.prompt = prompt
        print(f"Set prompt to '{self.prompt}'")

    def set_output_layout(self, 
                          output_path : str, 
                          width : Optional[int] = 768, 
                          height : Optional[int] = 432, 
                          fps : Optional[int] = 8, 
                          num_frames : Optional[int] = 24,
                          num_inference_steps : Optional[int] = 50) -> None:
        self.output_path = output_path
        self.video_size = [height, width]
        self.fps = fps
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        print(f"Set output path to '{self.output_path}'")
        print(f"Set video_size to {self.video_size}")
        print(f"Set video fps and num of frames to '{self.fps}' and '{self.num_frames}'")
        print(f"Set num of inference steps to '{self.num_inference_steps}'")

    def set_validation_images(self,
                  validation_image_start : str = None,
                  validation_image_end : Optional[str] = None):
        self.validation_image_start  = validation_image_start
        self.validation_image_end    = validation_image_end
        print(f"Set validation image of start to '{self.validation_image_start}'")
        print(f"Set validation image of end to '{self.validation_image_end}'")
