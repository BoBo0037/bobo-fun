import gc
import os
import logging
from pathlib import Path
import torch
import torchaudio
from typing import Optional
from src.mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video, setup_eval_logging)
from src.mmaudio.model.flow_matching import FlowMatching
from src.mmaudio.model.networks import MMAudio, get_my_mmaudio
from src.mmaudio.model.utils.features_utils import FeaturesUtils
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

class MMAudioManager():
    def __init__(self, 
                 device : torch.device, 
                 dtype : torch.dtype,
        ):
        self.device          : torch.device = device
        self.dtype           : torch.dtype = dtype
        self.variant         : str = "large_44k_v2" # small_16k, small_44k, medium_44k, large_44k, large_44k_v2
        self.output_dir      : str = "./output_mmaudio"
        self.prompt          : str = "puppy barking"
        self.negative_prompt : str = "music"
        self.video           : str = None
        self.num_steps       : int = 25
        self.duration        : float = 8.0
        self.cfg_strength    : float = 4.5
        self.seed            : int = None
        self.full_precision  : bool = True
        self.mask_away_clip  : bool = True
        self.skip_video_composite : bool = True

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()

    @torch.inference_mode()
    def generate(self):
        # logger
        setup_eval_logging()

        # set seed
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(2), "big")
        print(f"set seed to '{self.seed}'")

        if self.variant not in all_model_cfg:
            raise ValueError(f'Unknown model variant: {self.variant}')
        model: ModelConfig = all_model_cfg[self.variant]
        model.download_if_needed()
        seq_cfg = model.seq_cfg

        if self.video:
            video_path: Path = Path(self.video).expanduser()
        else:
            video_path = None

        self.dtype = torch.float32 if self.full_precision else torch.bfloat16

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # load a pretrained model
        net: MMAudio = get_my_mmaudio(model.model_name).to(self.device, self.dtype).eval()
        net.load_weights(torch.load(model.model_path, map_location=self.device, weights_only=True))
        log.info(f'Loaded weights from {model.model_path}')

        # misc setup
        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=self.num_steps)

        feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                    synchformer_ckpt=model.synchformer_ckpt,
                                    enable_conditions=True,
                                    mode=model.mode,
                                    bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                    need_vae_encoder=False)
        feature_utils = feature_utils.to(self.device, self.dtype).eval()

        if video_path is not None:
            log.info(f'Using video {video_path}')
            video_info = load_video(video_path, self.duration)
            clip_frames = video_info.clip_frames
            sync_frames = video_info.sync_frames
            self.duration = video_info.duration_sec
            if self.mask_away_clip:
                clip_frames = None
            else:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)
        else:
            log.info('No video provided -- text-to-audio mode')
            clip_frames = sync_frames = None

        seq_cfg.duration = self.duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        log.info(f'Prompt: {self.prompt}')
        log.info(f'Negative prompt: {self.negative_prompt}')

        audios = generate(clip_frames,
                        sync_frames, [self.prompt],
                        negative_text=[self.negative_prompt],
                        feature_utils=feature_utils,
                        net=net,
                        fm=fm,
                        rng=rng,
                        cfg_strength=self.cfg_strength)
        audio = audios.float().cpu()[0]
        if video_path is not None:
            save_path = self.output_dir + f'/{video_path.stem}.flac'
        else:
            safe_filename = self.prompt.replace(' ', '_').replace('/', '_').replace('.', '')
            save_path = self.output_dir + f'/{safe_filename}.flac'
        torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

        log.info(f'Audio saved to {save_path}')
        if video_path is not None and not self.skip_video_composite:
            video_save_path = self.output_dir + f'/{video_path.stem}.mp4'
            make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
            log.info(f'Video saved to {self.output_dir + video_save_path}')

        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))

    def set_output_layout(self, 
                          variant : Optional[str] = "large_44k_v2", 
                          prompt : Optional[str] = "puppy barking", 
                          negative_prompt : Optional[str] = "", 
                          video : Optional[str] = None, 
                          duration : Optional[float] = 8.0, 
                          num_steps : Optional[int] = 25) -> None:
        self.variant = variant
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.video = video
        self.duration = duration
        self.num_steps = num_steps
        print(f"Set variant to {self.variant}")
        print(f"Set prompt to {self.prompt}")
        print(f"Set negative_prompt to {self.negative_prompt}")
        print(f"Set video to {self.video}")
        print(f"Set duration to {self.duration}")
        print(f"Set num_steps to {self.num_steps}")
