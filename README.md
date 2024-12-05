# bobo-fun

bobo-fun is a project that collects interesting AI open-source projects that **run on a Mac** (All-In-One).

All code has been tested on a MacBook Pro **(M4 Max / 128GB RAM)**.

![Platform](https://img.shields.io/badge/platform-macOS-blue?style=flat-square)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/apache-2-0)

## Environment
To set up our environment, please run:
```sh
conda create --name bobo python=3.11
conda activate bobo
```
```sh
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
```
```sh
pip install -r requirements.txt
```
```sh
brew install ffmpeg
```

## Usage
Just run you want as belows

### commands

- [mflux](https://github.com/filipstrand/mflux)
```sh
python run_flux.py
```

- [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)
```sh
python run_glm.py
```

- [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
```sh
python run_phi.py
```

- [OmniGen](https://github.com/VectorSpaceLab/OmniGen)
```sh
python run_omnigen.py
```

- [CogVideo](https://github.com/THUDM/CogVideo)
```sh
python run_cogvideo.py
```

- [LTX-Video](https://github.com/Lightricks/LTX-Video)
```sh
python run_ltxvideo.py
```
- [mochi](https://github.com/genmoai/mochi)
```sh
python run_mochi.py    (black video bug)
```
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
```sh
python run_hyvideo.py  (black video bug)
```
- [suno-ai/bark](https://github.com/suno-ai/bark)
```sh
python run_suno_bark.py
```

## Issues:
1. CogVideoX(建议玩玩 CogVideoX-2b)。5B基本无法生成视频, 要么视频乱掉, 要么报错 total bytes of NDArray > 2**32 或 Invalid buffer size 问题...
2. LTX-Video(目前最适合在 mac 上玩), 能生成较长视频, 整体效果也还不错。 但极限长度情况下仍然有 2 中提到的报错问题...
3. Mochi(mac 上不推荐) 的 sample_model() ---> model_fn() ---> out_cond 和 out_uncond 模型采样值为 tensor([[nan, nan, nan,  ..., nan, nan, nan]...]), 导致输出全黑视频...
4. hyvideo 支持 mac 下 bfloat16 运行, 但视频生成结果是纯黑色...不知和 Mochi 是不是类似情况

## TODO:
- [x] support flux.1
- [x] support THUDM/glm-4-9b-chat
- [x] support microsoft/Phi-3.5-mini-instruct
- [x] support VectorSpaceLab/OmniGen
- [x] support THUDM/CogVideo
- [X] support Lightricks/LTX-Video
- [X] support genmoai/mochi
- [X] support Tencent/HunyuanVideo
- [X] support suno-ai/bark
- [ ] support GSeanCDAT/GIMM-VFI
<!-- - [ ] support k4yt3x/video2x -->
<!-- - [ ] support RVC-Boss/GPT-SoVITS -->
<!-- - [ ] support facebookresearch/audiocraft -->
<!-- - [ ] support haoheliu/AudioLDM2 -->
