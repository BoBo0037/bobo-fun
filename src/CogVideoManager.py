import os
import gc
import torch
from typing import Literal, Optional
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video
from utils.helper import check_and_make_folder

class CogVideoManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.prompt: str = "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."
        self.model_path: str = "THUDM/CogVideoX-2b"
        self.generate_type: str = "t2v"
        self.output_path: str = "output_cogvideox"
        self.width: int = 768
        self.height: int = 432
        self.fps: int = 8
        self.num_frames: int = 24
        self.guidance_scale: float = 6.5
        self.image_or_video_path: str = None
        self.num_inference_steps: int = 50
        self.num_videos_per_prompt: int = 1
        self.lora_path: str = None
        self.lora_rank: int = 128
        self.seed: int = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()

    def generate(self):
        # seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {self.seed}")

        # check output path
        check_and_make_folder(self.output_path)

        # get prefix
        index = len([path for path in os.listdir(self.output_path)]) + 1
        prefix = str(index).zfill(8)
        video_output_path = os.path.join(self.output_path, prefix + ".mp4")

        print("start generate video")
        generate_video(
            prompt=self.prompt,
            model_path=self.model_path,
            lora_path=self.lora_path,
            lora_rank=self.lora_rank,
            output_path=video_output_path,
            num_frames=self.num_frames,
            width=self.width,
            height=self.height,
            image_or_video_path=self.image_or_video_path,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_videos_per_prompt=self.num_videos_per_prompt,
            dtype=self.dtype,
            generate_type=self.generate_type,
            seed=self.seed,
            fps=self.fps
        )
        print("finish generate!")
    
    def set_model(self, 
                  model_path : str, 
                  generate_type : str) -> None:
        self.model_path = model_path
        self.generate_type = generate_type
        print(f"Set model path to '{self.model_path}'")
        print(f"Set video generate type to '{self.generate_type}'")

    def set_prompt(self, prompt : str) -> None:
        self.prompt = prompt
        print(f"Set prompt to '{self.prompt}'")

    def set_output_layout(self, 
                          output_path : str, 
                          width : Optional[int] = 680, 
                          height : Optional[int] = 384, 
                          fps : Optional[int] = 8, 
                          num_frames : Optional[int] = 81,
                          num_inference_steps : Optional[int] = 30) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        print(f"Set output path to '{self.output_path}'")
        print(f"Set video width and height to '{self.width}, {self.height}'")
        print(f"Set video fps and num of frames to '{self.fps}' and '{self.num_frames}'")
        print(f"Set num of inference steps to '{self.num_inference_steps}'")

    def set_input_image_or_video(self, image_or_video_path : str) -> None:
        self.image_or_video_path = image_or_video_path
        print(f"Set input image or video to '{self.image_or_video_path}'")


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 8,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    #pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    if torch.backends.mps.is_available():
        pipe.to("mps")

    else:
        pipe.to("cuda")
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=False,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,   # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    
    export_to_video(video_generate, output_path, fps=fps)
