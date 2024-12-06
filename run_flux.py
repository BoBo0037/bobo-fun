from src.FluxManager import FluxManager
from src.PromptManager import PromptManager

# set parameters
prompt = PromptManager("prompts.json").get("BMW")
total_num_imgs = 1
use_lora = False
use_img2img = False
use_controlnet = False
flux_manager = FluxManager()
flux_manager.set_model(model="schnell", quantize=8)
flux_manager.set_prompt(prompt=prompt)
flux_manager.set_output_layout(output="output_flux/img.png", width=720, height=480)

if use_lora:
    flux_manager.set_loras(
        lora_paths=[ "models/lora/F.1儿童简笔画风_v1.0.safetensors" ],
        lora_scales = [ 1.0 ],
        lora_triggers=[ "sketched style" ]
    )
if use_img2img: # cannot use img2img with controlnet
    flux_manager.set_img2img(
        init_image_path = "imgs/same_pose.png",
        init_image_strength = 0.3
    )
if use_controlnet:
    flux_manager.set_controlnet(
        controlnet_image_path = "imgs/skeletal2img.png",
        controlnet_save_canny = True,
        controlnet_strength = 1.0,
    )

# setup
flux_manager.setup(use_controlnet)

# generate images
flux_manager.generate(num_imgs = total_num_imgs, use_controlnet = use_controlnet)

# release
flux_manager.cleanup()
