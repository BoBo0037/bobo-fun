import gc
import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import TextStreamer

class GlmManager():
    def __init__(self, 
                 device : torch.device, 
                 dtype : torch.dtype,
                 model_name : str
        ):
        self.device = device
        self.dtype = dtype
        self.model_name=model_name

    def cleanup(self):
        print("Run cleanup")
        torch.cuda.empty_cache()
        gc.collect()

    def setup(self):
        print("Init tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name, 
            trust_remote_code=True
        )
        
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        print("Start setup model")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)

        print("Init pipeline")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
    
    @torch.inference_mode()
    def infer(self, query : Any, use_streamer : bool) -> str:
        print("Start inference")
        generated_text = self.pipe(
            text_inputs = query,
            streamer=self.streamer if use_streamer else None, 
            max_new_tokens = 256,
            temperature = 0.3,
            do_sample = True,
            return_full_text = False
        )
        return generated_text
