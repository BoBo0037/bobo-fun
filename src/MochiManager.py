import gc
import os
import torch
from typing import Optional
#from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from .genmo.lib.progress import progress_bar
from .genmo.lib.utils import save_video
from .genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

class MochiManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device     : torch.device = device
        self.dtype      : torch.dtype = dtype
        self.model      : str = "genmo/mochi-1-preview"
        self.model_cache : str = os.path.expanduser("~/.cache/huggingface/hub/models--genmo--mochi-1-preview")
        self.output_path: str = "./mochi_output.mp4"
        self.prompt     : str = "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."
        self.negative_prompt : str = ""
        self.width      : int = 848
        self.height     : int = 480
        self.num_frames : int = 31
        self.fps        : int = 6
        self.num_inference_steps : int = 64
        self.seed       : int = None
        
    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.cuda.empty_cache()

    def download_weights(self, model_cache : str):
        repo_id = self.model
        model = "dit.safetensors"
        decoder = "decoder.safetensors"
        encoder = "encoder.safetensors"

        if not os.path.exists(model_cache):
            print(f"Creating output directory: {model_cache}")
            os.makedirs(model_cache, exist_ok=True)

        def download_file(repo_id, model_cache, filename, description):
            file_path = os.path.join(model_cache, filename)
            if not os.path.exists(file_path):
                print(f"Downloading mochi {description} to: {file_path}")
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"*{filename}*"],
                    local_dir=model_cache,
                    local_dir_use_symlinks=False,
                )
            else:
                print(f"{description} already exists in: {file_path}")
            assert os.path.exists(file_path)

        download_file(repo_id, model_cache, decoder, "decoder")
        download_file(repo_id, model_cache, encoder, "encoder")
        download_file(repo_id, model_cache, model, "model")

    def setup(self):
        # check model cache
        print("Check model cache")
        self.download_weights(self.model_cache)

        # init pipeline
        print("Init pipeline")
        self.pipeline = MochiSingleGPUPipeline(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(model_path=f"{self.model_cache}/dit.safetensors", model_dtype="bf16"),
            decoder_factory=DecoderModelFactory(
                model_path=f"{self.model_cache}/decoder.safetensors",
            ),
            cpu_offload=False,
            #strict_load = False, 
            decode_type="tiled_full"
        )

    def generate(self):
        # seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {self.seed}")
    
        print("Start generate video")
        video = self.pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            sigma_schedule=linear_quadratic_schedule(64, 0.025),
            cfg_schedule=[4.5] * 64,
            batch_cfg=False
        )
        print("Save video")
        with progress_bar(type="tqdm"):
            save_video(video[0], self.output_path, fps=self.fps)
            #export_to_video(video[0], self.output_path, fps=self.fps)

    def set_prompt(self, 
                   prompt : str,
                   negative_prompt : Optional[str] = "worst quality, inconsistent motion, blurry, jittery, distorted") -> None:
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        print(f"Set prompt to '{self.prompt}'")
        print(f"Set negative prompt to '{self.negative_prompt}'")

    def set_output_layout(self, 
                          output_path : Optional[str] = "./mochi_output.mp4", 
                          width : Optional[int] = 848, 
                          height : Optional[int] = 480, 
                          num_frames : Optional[int] = 31,
                          fps : Optional[int] = 6,
                          num_inference_steps : Optional[int] = 64) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        print(f"Set output path to '{self.output_path}'")
        print(f"Set video width and height to '{self.width}, {self.height}'")
        print(f"Set video num of frames to '{self.num_frames}'")
        print(f"Set video fps to '{self.fps}'")
        print(f"Set num of inference steps to '{self.num_inference_steps}'")
