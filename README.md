# bobo-fun

bobo-fun is a project that collects interesting AI open-source projects that run on a Mac (All-In-One).

All code has been tested on a MacBook Pro (M4 Max / 128GB RAM).

![Platform](https://img.shields.io/badge/platform-macOS-blue?style=flat-square)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/apache-2-0)

## Environment
To set up our environment, please run:
```sh
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
```
```sh
pip install -r requirements.txt
```
```sh
brew install ffmpeg
```
```sh
pip install -U git+https://github.com/zRzRzRzRzRzRzR/diffusers.git
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

## Notes:
1. 目前 mac 上 cogvideo 似乎无法生成较长时间视频, 要么视频乱掉, 要么报错 total bytes of NDArray > 2**32 或 Invalid buffer size 问题
2. 目前 mac 上 LTX-Video 也有 total bytes of NDArray > 2**32 或 Invalid buffer size 问题

## TODO:
- [x] support flux.1 with filipstrand/mflux
- [x] support THUDM/glm-4-9b-chat
- [x] support microsoft/Phi-3.5-mini-instruct
- [x] support VectorSpaceLab/OmniGen
- [x] support THUDM/CogVideo
- [X] support Lightricks/LTX-Video
- [ ] support HVision-NKU/StoryDiffusion
- [ ] support suno-ai/bark
