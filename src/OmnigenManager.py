import os
import gc
import torch
from typing import List, Union, Optional
from src.OmniGen import OmniGenPipeline

class OmnigenManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.model               : str = "Shitao/OmniGen-v1"
        self.device     : torch.device = device
        self.dtype       : torch.dtype = dtype
        self.image                     = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()

    def setup(self):
        print("Init OmniGen pipe")
        self.pipe = OmniGenPipeline.from_pretrained(self.model)

    @torch.inference_mode()
    def generate(self,
                prompt : str = "a photograph of an astronaut riding a horse",
                output : Optional[str] = "omni_img.png", 
                input_images: Optional[Union[List[str], List[List[str]]]] = None, 
                width  : Optional[int] = 512, 
                height : Optional[int] = 512,
                seed   : Optional[int] = None, 
                num_inference_steps : Optional[int] = 16,
                guidance_scale      : Optional[float] = 2.5,
                img_guidance_scale  : Optional[float] = 1.6
        ) -> None: 
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        
        print("Start Text to Image")
        self.image = self.pipe(
            prompt=prompt, 
            input_images = input_images, 
            width=width, 
            height=height, 
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            img_guidance_scale = img_guidance_scale,
            dtype=self.dtype
        )
        # save
        self.image[0].save(output)
        print(f"Success: output image named {output}")
