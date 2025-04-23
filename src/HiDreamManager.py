import gc
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import UniPCMultistepScheduler, HiDreamImagePipeline
from utils.helper import check_and_make_folder

# Install diffusers:
# pip install git+https://github.com/huggingface/diffusers.git

class HiDreamManager:
    def __init__(self, device : torch.device, dtype : torch.dtype) -> None:
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        # "HiDream-ai/HiDream-I1-Fast"
        # "HiDream-ai/HiDream-I1-Dev"
        # "HiDream-ai/HiDream-I1-Full"
        self.model_id = "HiDream-ai/HiDream-I1-Dev"
        #self.model_id_llama3 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.model_id_llama3 = "/Users/zhangbo/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-8B-Instruct"
        self.prompt = 'A cat holding a sign that says "Hi-Dreams.ai", a realistic photo'
        self.output_path = "output_hidream"
        self.width = 512
        self.height = 512
        self.guidance_scale=5.0
        self.num_inference_steps=28
    
    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()
    
    def setup(self) -> None:
        # check output folder
        check_and_make_folder(self.output_path)

        print("setup tokenizer")
        self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(self.model_id_llama3)

        print("setup text encoder")
        self.text_encoder_4 = LlamaForCausalLM.from_pretrained(
            self.model_id_llama3,
            torch_dtype=self.dtype,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        print("setup hidream pipeline")
        self.pipe = HiDreamImagePipeline.from_pretrained(
            self.model_id,
            tokenizer_4=self.tokenizer_4,
            text_encoder_4=self.text_encoder_4,
            torch_dtype=self.dtype,
            return_dict_in_generate=True,
            output_attentions=True
        )
        self.pipe.to(self.device)
    
    @torch.inference_mode()
    def generate(self):
        print("start generate image")
        image = self.pipe(
            prompt=self.prompt,
            width=self.width,
            height=self.height,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator("mps").manual_seed(0)
        ).images[0]
        image.save(self.output_path + "/output.png")
    
    def set_prompt(self, prompt : str) -> None:
        self.prompt = prompt
        print(f"Set prompt to '{self.prompt}'")

    def set_output_layout(self, width : int, height : int) -> None:
        self.width = width
        self.height = height
        print(f"Set image width and height to '{self.width}, {self.height}'")
    