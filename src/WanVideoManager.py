import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers.models.transformers.transformer_wan as transformer_wan
from typing import Optional,Tuple
from transformers import CLIPVisionModel, UMT5EncoderModel
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils import export_to_video, load_image
from utils.helper import check_and_make_folder

class WanVideoManager():
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype : torch.dtype = dtype
        # Available models: 
        # T2V: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        # I2V: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
        self.model_id_t2v = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        self.model_id_i2v = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        self.prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        self.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        self.output_path = "output_wanvideo"
        self.image = None
        self.width = 832
        self.height = 480
        self.num_frames = 25
        self.fps = 8
        self.num_inference_steps = 30
        self.guidance_scale = 5.0
        self.flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        self.max_area = 720 * 832
        self.seed = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()

    def setup(self, enable_i2v : bool):
        # set seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        print(f"set seed to '{self.seed}'")
        
        # check output folder
        check_and_make_folder(self.output_path)

        print("setup vae")
        self.vae = AutoencoderKLWan.from_pretrained(self.model_id_t2v, subfolder="vae", torch_dtype=torch.float32)

        print("setup scheduler")
        self.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=self.flow_shift)
        
        if enable_i2v:
            print("setup text encoder")
            self.text_encoder = UMT5EncoderModel.from_pretrained(self.model_id_i2v, subfolder="text_encoder", torch_dtype=self.dtype)
            
            print("setup image encoder")
            self.image_encoder = CLIPVisionModel.from_pretrained(self.model_id_i2v, subfolder="image_encoder", torch_dtype=torch.float32)
            
            print("setup transformer")
            self.transformer = WanTransformer3DModel.from_pretrained(self.model_id_i2v, subfolder="transformer", torch_dtype=self.dtype)
            self.transformer.enable_layerwise_casting(storage_dtype=torch.bfloat16, compute_dtype=self.dtype)
            
            print("setup Wan video image2video pipeline")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id_i2v, 
                text_encoder=self.text_encoder, 
                image_encoder=self.image_encoder, 
                vae=self.vae, 
                transformer=self.transformer, 
                torch_dtype=self.dtype)
        else:
            print("setup Wan video pipeline")
            self.pipe = WanPipeline.from_pretrained(self.model_id_t2v, vae=self.vae, torch_dtype=self.dtype)
        
        self.pipe.scheduler = self.scheduler
        self.pipe.to(self.device)

        if enable_i2v and self.image is not None:
            print("process first frame image...")
            self.first_frame = load_image(self.image)
            aspect_ratio = self.first_frame.height / self.first_frame.width
            mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
            height = round(np.sqrt(self.max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(self.max_area / aspect_ratio)) // mod_value * mod_value
            self.first_frame = self.first_frame.resize((width, height))

    @torch.inference_mode()
    def generate(self, enable_i2v : bool):
        if enable_i2v:
            print("start image to video process")
            output = self.pipe(
                image=self.first_frame,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                num_inference_steps = self.num_inference_steps
            ).frames[0]
        else:
            print("start text to video process")
            output = self.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                num_inference_steps = self.num_inference_steps
            ).frames[0]

        print("start save video")
        index = len([path for path in os.listdir(self.output_path)]) + 1
        prefix = str(index).zfill(8)
        video_name = os.path.join(self.output_path, prefix + ".mp4")
        export_to_video(output, video_name, fps=self.fps)

    def set_prompt(self, prompt : str) -> None:
            self.prompt = prompt
            print(f"Set prompt to '{self.prompt}'")

    def set_output_layout(self, 
                          output_path : Optional[str] = "output_wanvideo", 
                          width : Optional[int] = 832, 
                          height : Optional[int] = 480, 
                          fps : Optional[int] = 8, 
                          num_frames : Optional[int] = 25,
                          num_inference_steps : Optional[int] = 30
                          ) -> None:
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

    def set_image(self, input_image : str) -> None:
        self.image = input_image
        print(f"Set image to '{self.image}'")

# change freqs_dtype from torch.float64 to torch.float32
class WanRotaryPosEmbedMPS(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float32
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

def replace_wan_transformer():
    # define mps call
    def __call__mps(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            # here，using float32
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float32).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
    
    # backup
    original_class = transformer_wan.WanRotaryPosEmbed
    original_call = transformer_wan.WanAttnProcessor2_0.__call__

    # replace
    transformer_wan.WanRotaryPosEmbed = WanRotaryPosEmbedMPS
    transformer_wan.WanAttnProcessor2_0.__call__ = __call__mps
    
    return original_class, original_call

def revert_wan_transformer(original_class, original_call):
    transformer_wan.WanRotaryPosEmbed = original_class
    transformer_wan.WanAttnProcessor2_0.__call__ = original_call
