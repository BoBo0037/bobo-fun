import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import subprocess
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from src.hyvideo.utils.file_utils import save_videos_grid
from src.hyvideo.config import parse_args
from src.hyvideo.inference import HunyuanVideoSampler
from src.hyvideo.config import MODEL_BASE
from src.PromptManager import PromptManager

# Resolution	       h/w=9:16	         h/w=16:9    	    h/w=4:3	        h/w=3:4	         h/w=1:1
#  540p	            544px960px129f	  960px544px129f    624px832px129f	 832px624px129f	  720px720px129f
#  720p(recommend)	720px1280px129f	  1280px720px129f	1104px832px129f	 832px1104px129f  960px960px129f

def main():
    # set args
    args = parse_args()
    args.model_base = MODEL_BASE
    args.dit_weight = MODEL_BASE + "/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
    args.save_path = "hyvideo-output"
    args.prompt = PromptManager("prompts.json").get("suv")
    args.neg_prompt = PromptManager("prompts.json").get("negative-video")
    args.video_size=(624, 832)
    args.video_fps = 8
    args.video_length = 17
    args.infer_steps = 50
    args.disable_autocast=True
    args.seed = int.from_bytes(os.urandom(2), "big")
    # precisions
    args.precision="bf16"
    args.text_encoder_precision="bf16"
    args.text_encoder_precision_2="bf16"
    args.vae_precision="bf16"
    print(f"Set args to: '{args}'")

    # check model weights
    check_model(MODEL_BASE)

    # model root path
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    for i, sample in enumerate(samples):
        sample = samples[i].unsqueeze(0)
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
        save_videos_grid(sample, save_path, fps=args.video_fps)
        logger.info(f'Sample save to: {save_path}')


def check_model(model_dir : str) -> None: 
    # HunyuanVideo
    #   ├──ckpts
    #   │  ├──README.md
    #   │  ├──hunyuan-video-t2v-720p
    #   │  │  ├──transformers
    #   ├  │  ├──vae
    #   │  ├──text_encoder
    #   │  ├──text_encoder_2
    #   ├──...

    # download hunyuanVideo ckpt
    command_download_tencent_HunyuanVideo = [
        "huggingface-cli",
        "download",
        "tencent/HunyuanVideo",
        "--local-dir",
        model_dir
    ]
    subprocess.run(command_download_tencent_HunyuanVideo, env=os.environ)

    # download text encoder
    command_download_llava_llama_3_8b_v1_1_transformers = [
        "huggingface-cli",
        "download",
        "xtuner/llava-llama-3-8b-v1_1-transformers",
        "--local-dir",
        model_dir + "/llava-llama-3-8b-v1_1-transformers"
    ]
    subprocess.run(command_download_llava_llama_3_8b_v1_1_transformers, env=os.environ)

    # download text encoder 2
    command_download_clip_vit_large_patch14 = [
        "huggingface-cli",
        "download",
        "openai/clip-vit-large-patch14",
        "--local-dir",
        model_dir + "/text_encoder_2"
    ]
    subprocess.run(command_download_clip_vit_large_patch14, env=os.environ)

    # separate the language model parts of llava-llama-3-8b-v1_1-transformers into text_encoder
    command_separate_language_model_parts = [
        "python",
        "src/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py",
        "--input_dir", 
        model_dir + "/llava-llama-3-8b-v1_1-transformers",
        "--output_dir",
        model_dir + "/text_encoder"
    ]
    subprocess.run(command_separate_language_model_parts, env=os.environ)


if __name__ == "__main__":
    main()
