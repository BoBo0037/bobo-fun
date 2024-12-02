import os
os.environ["SUNO_ENABLE_MPS"] = "1"
os.environ["SUNO_OFFLOAD_CPU"] = "1"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
import scipy
import torch
from transformers import AutoProcessor, BarkModel
from IPython.display import Audio
from utils.helper import set_device

MODEL = "suno/bark" # "suno/bark-small"

# set device
device = set_device()
dtype = torch.float32

# model
model = BarkModel.from_pretrained(MODEL, torch_dtype=dtype)
model = model.to_bettertransformer()
model = model.to('cpu') # 'mps' error ...

# processor
processor = AutoProcessor.from_pretrained(MODEL)

# known non-speech sounds:
# [laughter]
# [laughs]
# [sighs]
# [music]
# [gasps]
# [clears throat]
# — or ... for hesitations
# ♪ for song lyrics
# CAPITALIZATION for emphasis of a word
# [MAN] and [WOMAN] to bias Bark toward male and female speakers, respectively

# english
inputs = processor("Hello, my name is Suno. And, uh - and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.", voice_preset="v2/en_speaker_6")

# chinese
#inputs = processor("在晨曦微露的清晨，静谧的湖面上泛起层层涟漪，仿佛是大自然在轻声诉说着昨夜的梦境。微风拂过，带来一丝清凉，也带来了新一天的希望与期待。此刻，时间仿佛放慢了脚步，让人得以在这片刻的宁静中，细细品味生活的点滴美好。", voice_preset="v2/zh_speaker_8")

# generate music
#inputs = processor("♪ In the jungle, the mighty jungle, the lion barks tonight ♪")

# audio array
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# sample rate
sample_rate = model.generation_config.sample_rate

# listen
#Audio(audio_array, rate=sample_rate)

# save
scipy.io.wavfile.write("bark_output.wav", rate=sample_rate, data=audio_array)
