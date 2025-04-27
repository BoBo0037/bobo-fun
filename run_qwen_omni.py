import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
import soundfile as sf
from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    Qwen2_5OmniThinkerForConditionalGeneration, 
    Qwen2_5OmniProcessor
)
from utils.helper import set_device, check_and_make_folder

# more examples:
# https://github.com/huggingface/transformers/releases/tag/v4.51.3-Qwen2.5-Omni-preview

# settings
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
AUDIO_OUTPUT_PATH = "output_qwen_omni/"

# check output folder
check_and_make_folder(AUDIO_OUTPUT_PATH)

# init
device = set_device()

# prepare conversations
conversation_1 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What cant you hear and see in this video? What is the subtitle content?"},
            {"type": "video", "video": "assets/vids/onepiece_demo.mp4"},
            #{"type": "image", "path": "/path/to/image.jpg"},
            #{"type": "audio", "path": "/path/to/audio.wav"},
        ],
    },
]
conversations = [ conversation_1 ]

print("start setup model...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    enable_audio_output=True
)
model = model.to(device)

print("start setup processor...")
# set image range
#min_pixels = 128 * 28 * 28
#max_pixels = 768 * 28 * 28
processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_ID, 
    #min_pixels=min_pixels,
    #max_pixels=max_pixels
)

print("prepare imputs...")
inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_fps=1,
    # kwargs to be passed to `Qwen2-5-OmniProcessor`
    padding=True,
    use_audio_in_video=True
).to(model.device)

# Generation params for audio or text can be different and have to be prefixed with `thinker_` or `talker_`
print("start generate...")
text_ids, audio = model.generate(
    **inputs, 
    use_audio_in_video=True, 
    thinker_do_sample=False, 
    talker_do_sample=True,
    return_audio=True,
    #spk="Chelsie",
)
text = processor.batch_decode(
    text_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)

print("write audio and text answer...")
sf.write(
    AUDIO_OUTPUT_PATH + "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
print(text)
print("finish...")