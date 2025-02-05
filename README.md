# bobo-fun

bobo-fun is a project that collects interesting AI open-source projects that **run on a Mac** (All-In-One).

All code has been tested on a MacBook Pro **(M4 Max / 128GB RAM)**.

![Platform](https://img.shields.io/badge/platform-macOS-blue?style=flat-square)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/apache-2-0)

## News:
- Support [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) with low memory using

  Using the HunyuanVideoPipeline from the newly released diffusers 0.32.2, everything runs well on a Mac...

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
- [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion)
```sh
python run_story_diffusion.py
```
- [FlowEdit](https://github.com/fallenshock/FlowEdit)
```sh
python run_flow_edit.py
```
- [RMBG](https://huggingface.co/briaai/RMBG-2.0)
```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 python run_rmbg.py
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
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
```sh
python run_hyvideo.py
```
- [Tencent/MimicMotion](https://github.com/Tencent/MimicMotion)
```sh
conda remove pytorch torchvision torchaudio
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
pip install -r requirements.txt
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
- [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
```sh
python run_kokoro_82m.py
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

2. Mochi 目前有纯黑视频 bug。

   Mochi: sample_model() ---> model_fn() ---> out_cond 和 out_uncond 模型采样值为 tensor([[nan, nan, ..., nan, nan]...]), 导致纯黑问题。

## Supports:
- [x] 1. THUDM/glm-4-9b-chat               (文生文)
- [x] 2. microsoft/Phi-3.5-mini-instruct   (文生文)
- [x] 3. flux.1                            (文生图)
- [x] 4. VectorSpaceLab/OmniGen            (文生图)
- [X] 5. HVision-NKU/StoryDiffusion        (文生图, 擅长故事)
- [X] 6. fallenshock/FlowEdit              (修图)
- [X] 7. briaai/RMBG-2.0                   (图片背景剔除)
- [x] 8. THUDM/CogVideo                    (文生视频)
- [x] 9. aigc-apps/CogVideoX-Fun           (文生视频)
- [X] 10. Lightricks/LTX-Video              (文生视频)
- [X] 11. genmoai/mochi                    (文生视频)
- [X] 12. Tencent/HunyuanVideo             (文生视频)
- [X] 13. Tencent/MimicMotion              (动作/跳舞)
- [X] 14. xinntao/Real-ESRGAN              (视频超分)
- [X] 15. hzwer/ECCV2022-RIFE              (视频插帧)
- [X] 16. thewh1teagle/kokoro-onnx         (文生音频)
- [X] 17. suno-ai/bark                     (文生音频)
- [X] 18. hkchengrex/MMAudio               (文生音频, 擅长音效/音乐)
