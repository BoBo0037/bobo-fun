import os
import gc
import torch
from typing import Optional
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models import AutoencoderKLHunyuanVideo
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from utils.helper import check_and_make_folder

class HunyuanVideoManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.model_id = "hunyuanvideo-community/HunyuanVideo"
        self.prompt = "A cat walks on the grass, realistic"
        self.output_path = "output_hyvideo"
        self.width = 512
        self.height = 320
        self.num_frames = 31
        self.fps = 15
        self.num_inference_steps = 30
        self.guidance_scale = 6.0
        self.seed = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()
    
    def delete_pipe(self, pipe):
        if hasattr(pipe, 'scheduler'): del pipe.scheduler
        if hasattr(pipe, 'transformer'): del pipe.transformer
        if hasattr(pipe, 'vae'): del pipe.vae
        if hasattr(pipe, 'text_encoder'): del pipe.text_encoder
        if hasattr(pipe, 'text_encoder_2'): del pipe.text_encoder_2
        if hasattr(pipe, 'tokenizer'): del pipe.tokenizer
        if hasattr(pipe, 'tokenizer_2'): del pipe.tokenizer_2
        del pipe

    def flush(self, pipe):
        print("flush gpu memory")
        self.delete_pipe(pipe)
        self.cleanup()

    def setup(self):
        # set seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        print(f"set seed to '{self.seed}'")
        
        # check output folder
        check_and_make_folder(self.output_path)

        print("start setup 8-bit transformer")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            self.model_id, 
            subfolder="transformer", 
            torch_dtype=self.dtype
        )

        print("start setup hyvideo pipeline")
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            self.model_id, 
            transformer=transformer, 
            torch_dtype=torch.float16
        )
        self.pipe.to(self.device)

    def setup_low_mem(self):
        # set seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        print(f"set seed to '{self.seed}'")
        
        # check output folder
        check_and_make_folder(self.output_path)

    @torch.inference_mode()
    def generate(self):
        print("start generate video")
        output = self.pipe(
            prompt=self.prompt,
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).frames[0]

        print("start save video")
        index = len([path for path in os.listdir(self.output_path)]) + 1
        prefix = str(index).zfill(8)
        video_name = os.path.join(self.output_path, prefix + ".mp4")
        export_to_video(output, video_name, fps=self.fps)
    
    @torch.inference_mode()
    def generate_low_mem(self):
        print("Load tokenizer and text_encoder only")
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            transformer=None,
            vae=None,
        )
        self.pipe = self.pipe.to(self.device)

        print("Encoding prompts")
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
                prompt=self.prompt, 
                prompt_2=None, 
                device=self.device
            )
        # release
        self.flush(self.pipe)

        print("Load hunyuan pipeline with transformer only")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            self.model_id, 
            subfolder="transformer", 
            torch_dtype=self.dtype
        )
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            transformer=transformer,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=None,
        )
        self.pipe = self.pipe.to(self.device)

        print("start generate video latents")
        with torch.no_grad():
            latents = self.pipe(
                output_type="latent", 
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask, 
                width=self.width, 
                height=self.height, 
                num_frames=self.num_frames, 
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator("cpu").manual_seed(self.seed)
            ).frames
        # release
        self.flush(self.pipe)

        print("load vae of hunyuan")
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to(self.device)

        print("start decode latents")
        with torch.no_grad():
            # decode latents
            latents = latents.to(vae.dtype) / vae.config.scaling_factor
            video = vae.decode(latents, return_dict=False)[0]
            # video Process
            vae_scale_factor_spatial = (vae.spatial_compression_ratio if vae is not None else 8)
            video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
            video = video_processor.postprocess_video(video, output_type="pil")
        
        print("start save video")
        index = len([path for path in os.listdir(self.output_path)]) + 1
        prefix = str(index).zfill(8)
        video_name = os.path.join(self.output_path, prefix + ".mp4")
        export_to_video(video[0], video_name, fps=self.fps)

    def set_prompt(self, prompt : str) -> None:
            self.prompt = prompt
            print(f"Set prompt to '{self.prompt}'")

    def set_output_layout(self, 
                          output_path : Optional[str] = "output_hyvideo", 
                          width : Optional[int] = 512, 
                          height : Optional[int] = 320, 
                          fps : Optional[int] = 8, 
                          num_frames : Optional[int] = 25,
                          num_inference_steps : Optional[int] = 35) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        print(f"Set output path to '{self.output_path}'")
        print(f"Set video [width, height] to [{self.width}, {self.height}]")
        print(f"Set video fps and num of frames to '{self.fps}' and '{self.num_frames}'")
        print(f"Set num of inference steps to '{self.num_inference_steps}'")
