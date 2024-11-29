import os
import time
import threading
import gc
from pathlib import Path
from mflux import Flux1, Config, ModelConfig
from mflux import Flux1Controlnet, ConfigControlnet
from mflux import StopImageGenerationException
from utils.helper import get_new_object_name_with_index, get_new_folder_name_with_index
from utils.helper import check_and_init_folder, find_single_file_with_suffix, remove_files_except_with_suffix
from utils.helper import show_img, calc_time_consumption

class FluxManager:
    def __init__(self) -> None:
        # args
        self.prompt       : str = "a photograph of an astronaut riding a horse"
        self.final_prompt : str = self.prompt
        self.model        : str = "schnell" # "dev" or "schnell"
        self.output       : str = "img.png"
        self.quantize     : int = 8
        self.width        : int = 512
        self.height       : int = 512
        self.num_inference_steps : int = 4 if self.model == "schnell" else 16
        self.guidance     : float = 3.5
        self.seed         : int = None
        self.local_path   : str = None
        self.metadata     : bool = False
        self.stepwise_output_dir : str = "output_stepwise"
        self.stepwise_composite_suffix : str = "_composite.png"
        # lora args
        self.lora_paths    : list[str] = None
        self.lora_scales   : list[float] = None
        self.lora_triggers : list[str] = None
        # img2img args
        self.init_image_path     : str = None
        self.init_image_strength : float = 0.3
        # controlnet args
        self.controlnet_image_path : str = None
        self.controlnet_save_canny : bool = False
        self.controlnet_strength   : float = 1.0
        # others
        self.image                = None
        self.flux_pipe            = None
        self.flux_controlnet_pipe = None
        self.last_file_size       = None
        self.stop_event = threading.Event()
        self.start_time : float = 0
        self.end_time   : float = 0
    
    def cleanup(self)-> None:
        print("Run cleanup")
        gc.collect()
        
    def setup(self, use_controlnet : bool) -> None:
        self.start_time = time.time()
        common_params = {
            'model_config': ModelConfig.from_alias(self.model),
            'quantize': self.quantize,
            'local_path': self.local_path,
            'lora_paths': self.lora_paths,
            'lora_scales': self.lora_scales,
        }
        if use_controlnet:
            print("Loading flux controlnet pipe")
            self.flux_controlnet_pipe = Flux1Controlnet(**common_params)
        else: 
            print("Loading flux pipe")
            self.flux_pipe = Flux1(**common_params)

    def generate(self, num_imgs: int, use_controlnet : bool) -> None:
        print(f"Start generate {num_imgs} images entirely")
        try: 
            for index in range(num_imgs):
                # prepare folder of stepwise images
                stepwise_img_folder = get_new_folder_name_with_index(self.stepwise_output_dir, index)
                check_and_init_folder(stepwise_img_folder)
                # launch a thread to show stepwise images
                self.stop_event.clear()
                thread_name = f"stepwise_image_thread_{index}"
                thread = threading.Thread(target=lambda: self.show_stepwise_img(stepwise_img_folder, 0.5), name=thread_name)
                thread.start()
                # generate single image
                self.generate_img(stepwise_img_folder, index, use_controlnet)
                # save image
                output_path_with_index = get_new_object_name_with_index(self.output, index)
                self.save_img(output_path_with_index, index)
                # stop thread
                self.stop_event.set()
                thread.join()
                # remove all files except '_composite.png' suffix file in stepwise image folder
                remove_files_except_with_suffix(stepwise_img_folder, self.stepwise_composite_suffix)
        except StopImageGenerationException as stop_exc:
            print(f"Failed to generate image: {stop_exc}")
        except KeyboardInterrupt as keyInterrupt:
            print(f"Failed to generate image: {keyInterrupt}")
        except Exception as error:
            print(f"Failed to generate image: {error}")
        finally:
            self.stop_event.set()
            self.end_time = time.time()
            calc_time_consumption(self.start_time, self.end_time)
            self.cleanup()
            print("Finish all")

    def generate_img(self, stepwise_img_folder : str, index : int, use_controlnet : bool) -> None:
        print(f"Generating {index}-th image")
        # set pip and config class
        pipe_class, config_class = self.get_pipe_and_config(use_controlnet)
        # init params
        params = {
            'seed': int(time.time()) if self.seed is None else self.seed,
            'prompt': self.final_prompt,
            'stepwise_output_dir': Path(stepwise_img_folder),
        }
        # update params and configs
        configs = {
            'num_inference_steps': self.num_inference_steps,
            'width': self.width,
            'height': self.height,
            'guidance': self.guidance,
        }
        if use_controlnet:
            params.update({ 'output': self.output })
            params.update({ 'controlnet_image_path': self.controlnet_image_path })
            params.update({ 'controlnet_save_canny': self.controlnet_save_canny })
            final_config = config_class(**configs, 
                             controlnet_strength=self.controlnet_strength)
        else:
            final_config = config_class(**configs, 
                             init_image_path=Path(self.init_image_path) if self.init_image_path else None, 
                             init_image_strength=self.init_image_strength)
        # add configs to params
        params.update({'config': final_config})
        # generate image
        self.image = pipe_class.generate_image(**params)
        
    def save_img(self, output_path: str, index : int) -> None:
        print(f"Saving {index}-th image to {output_path}")
        self.image.save(path=output_path, export_json_metadata=self.metadata)

    def get_pipe_and_config(self, use_controlnet: bool) -> tuple:
        if use_controlnet:
            if self.flux_controlnet_pipe is None:
                raise ValueError("flux controlnet pipe is None, please run load_model() first")
            return self.flux_controlnet_pipe, ConfigControlnet
        else:
            if self.flux_pipe is None:
                raise ValueError("flux pipe is None, please run load_model() first")
            return self.flux_pipe, Config

    def set_prompt(self, prompt : str) -> None:
        self.prompt = prompt
        self.final_prompt = self.update_final_prompt()

    def set_model(self, model : str, quantize : int) -> None:
        self.model = model
        self.quantize = quantize
        print(f"Set model to '{self.model}'")
        print(f"Set quantize to '{self.quantize}'")

    def set_output_layout(self, output : str, width : int, height : int) -> None:
        self.output = output
        self.width = width
        self.height = height
        print(f"Set output to '{self.output}'")
        print(f"Set image width and height to '{self.width}, {self.height}'")
        
    def set_loras(self, lora_paths : list[str], lora_scales : list[float], lora_triggers : list[str]) -> None:
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.lora_triggers = lora_triggers
        print(f"Set lora path to '{self.lora_paths}'")
        print(f"Set lora scales to '{self.lora_scales}'")
        print(f"Set lora trigger to '{self.lora_triggers}'")
        self.final_prompt = self.update_final_prompt()

    def set_img2img(self, init_image_path : str, init_image_strength : float) -> None:
        self.init_image_path = init_image_path
        self.init_image_strength = init_image_strength
        print(f"Set init image path to '{self.init_image_path}'")
        print(f"Set init image strength to '{self.init_image_strength}'")

    def set_controlnet(self, controlnet_image_path : str, controlnet_save_canny : bool, controlnet_strength : float) -> None:
        self.controlnet_image_path = controlnet_image_path
        self.controlnet_save_canny = controlnet_save_canny
        self.controlnet_strength = controlnet_strength
        print(f"Set controlnet image path to '{self.controlnet_image_path}'")
        print(f"Set controlnet save canny to '{self.controlnet_save_canny}'")
        print(f"Set controlnet strength to '{self.controlnet_strength}'")
        
    def update_final_prompt(self) -> str:
        if self.lora_triggers is None:
            final_prompt = self.prompt
        else:
            final_prompt = ", ".join(self.lora_triggers) + ", " + self.prompt
        print(f"Final prompt is: '{final_prompt}'")
        return final_prompt

    def show_stepwise_img(self, stepwise_img_folder : str, seconds : float) -> None:
        while not self.stop_event.is_set():
            self.show_composite_stepwise_img(stepwise_img_folder)
            time.sleep(seconds)

    def show_composite_stepwise_img(self, stepwise_img_folder : str) -> None:
        file_path = find_single_file_with_suffix(stepwise_img_folder, self.stepwise_composite_suffix)
        if file_path is not None:
            file_size = os.path.getsize(file_path)
            if self.last_file_size is None or self.last_file_size != file_size:
                show_img(file_path, "stepwise composite img")
            self.last_file_size = file_size
