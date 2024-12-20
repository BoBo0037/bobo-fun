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
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
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
- [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun)
```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 python run_cogvideofun.py
```
- [LTX-Video](https://github.com/Lightricks/LTX-Video) with [STGuidance](https://github.com/junhahyung/STGuidance)
```sh
python run_ltxvideo.py
```
- [mochi](https://github.com/genmoai/mochi) ---------> **PS. black video bug**
```sh
python run_mochi.py
```
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) ---------> **PS. black video bug**
```sh
python run_hyvideo.py
```
- [Tencent/MimicMotion](https://github.com/Tencent/MimicMotion)
```sh
python run_mimic_motion.py
```
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
```sh
python run_realesrgan.py
```
- [hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)
```sh
python run_rife.py
```
- [suno-ai/bark](https://github.com/suno-ai/bark)
```sh
python run_suno_bark.py
```
- [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio)
```sh
python run_mmaudio.py
```

## Issues:
1. 目前几乎所有文生视频项目, 当分辨率或生成帧数较大时, 都有 total bytes of NDArray > 2**32 或 Invalid buffer size 报错问题。这似乎是 mac 本身内部实现问题, 暂无太好方法。

2. CogVideoX 系列整体效果比较好。 但目前 mac 上只建议 2b 模型, 5b 版有上述 1 中问题。CogVideoX-Fun 版类似, 整体更灵活, 动作幅度更大, 两者生成质量其实总体差不多, 各有优缺点。

2. LTX-Video-2b 整体效果稍弱于 CogVideoX-2b。但它有生成速度快, 生成视频时间长, 支持多种模式等优点, 是目前最适合在 mac 上玩的文生视频模型, 非常期待后续更新。

3. Mochi 和 hyvideo(效果应该是当下开源中最好) 目前均有纯黑视频 bug, 暂时没找到好方法解决。

   Mochi: sample_model() ---> model_fn() ---> out_cond 和 out_uncond 模型采样值为 tensor([[nan, nan, ..., nan, nan]...]), 导致纯黑问题。

   hyvideo: spatial_tiled_decode() -> decoded = self.decoder(tile) -> sample = self.conv_in(sample) 这里输入 sample 后得出全 nan 的 值, 导致纯黑问题。

## Supports:
- [x] support THUDM/glm-4-9b-chat               (文生文)
- [x] support microsoft/Phi-3.5-mini-instruct   (文生文)
- [x] support flux.1                            (文生图)
- [x] support VectorSpaceLab/OmniGen            (文生图)
- [x] support THUDM/CogVideo                    (文生视频)
- [x] support aigc-apps/CogVideoX-Fun           (文生视频)
- [X] support Lightricks/LTX-Video              (文生视频)
- [X] support genmoai/mochi                     (文生视频)
- [X] support Tencent/HunyuanVideo              (文生视频)
- [X] support Tencent/MimicMotion               (动作/跳舞)
- [X] support xinntao/Real-ESRGAN               (视频超分)
- [X] support hzwer/ECCV2022-RIFE               (视频插帧)
- [X] support suno-ai/bark                      (文生音频, 擅长语音/对话)
- [X] support hkchengrex/MMAudio                (文生音频, 擅长音效/音乐)
