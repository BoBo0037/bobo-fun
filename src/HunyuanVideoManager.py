import os
import gc
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
#from utils.helper import check_and_make_folder

class HunyuanVideoManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.model_id = "hunyuanvideo-community/HunyuanVideo"
        self.prompt = "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it's tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."
        #self.negative_prompt = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."
        self.output_path = "output_hyvideo"
        self.width = 512
        self.height = 320
        self.num_frames = 9
        self.fps = 8
        self.num_inference_steps = 15
        self.guidance_scale = 6.5
        self.seed = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()
    
    def setup(self):
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
        export_to_video(output, "output.mp4", fps=self.fps)
