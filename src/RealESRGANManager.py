import gc
import os
import cv2
import glob
import argparse
import mimetypes
import numpy as np
import shutil
import subprocess
import torch
import ffmpeg
from os import path as osp
from gfpgan import GFPGANer
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from src.realesrgan.utils import RealESRGANer
from src.realesrgan.archs.srvgg_arch import SRVGGNetCompact

class RealESRGANManager:
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.model = None
        self.netscale = None
        self.file_url = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()

    def process_img(self, args):
        print("Start a image process!")
        print(f"Set args to: '{args}'")
        # determin model
        self.determine_model(args)
        # determin model path
        model_path = self.determine_model_path(args)
        # use dni to control the denoise strength
        dni_weight = None
        if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
        # restorer
        upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=self.model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=not args.fp32,
            gpu_id=args.gpu_id,
            device=self.device
        )
        if args.face_enhance:  # Use GFPGAN for face enhancement
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        os.makedirs(args.output, exist_ok=True)

        if os.path.isfile(args.input):
            paths = [args.input]
        else:
            paths = sorted(glob.glob(os.path.join(args.input, '*')))

        for idx, path in enumerate(paths):
            imgname, extension = os.path.splitext(os.path.basename(path))
            print('Testing', idx, imgname)

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            else:
                img_mode = None

            try:
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                if args.ext == 'auto':
                    extension = extension[1:]
                else:
                    extension = args.ext
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                if args.suffix == '':
                    save_path = os.path.join(args.output, f'{imgname}.{extension}')
                else:
                    save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
                cv2.imwrite(save_path, output)

    def process_video(self, args):
        print("Start a video process!")
        print(f"Set args to: '{args}'")

        args.input = args.input.rstrip('/').rstrip('\\')
        os.makedirs(args.output, exist_ok=True)

        if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
            is_video = True
        else:
            is_video = False

        if is_video and args.input.endswith('.flv'):
            mp4_path = args.input.replace('.flv', '.mp4')
            os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
            args.input = mp4_path

        if args.extract_frame_first and not is_video:
            args.extract_frame_first = False

        self.run_video(args)

        if args.extract_frame_first:
            tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
            shutil.rmtree(tmp_frames_folder)

    def run_video(self, args):
        args.video_name = osp.splitext(os.path.basename(args.input))[0]
        video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

        if args.extract_frame_first:
            tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
            os.makedirs(tmp_frames_folder, exist_ok=True)
            os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {tmp_frames_folder}/frame%08d.png')
            args.input = tmp_frames_folder

        num_gpus = torch.cuda.device_count() if not torch.backends.mps.is_available() else 1
        num_process = num_gpus * args.num_process_per_gpu
        if num_process == 1:
            self.inference_video(args, video_save_path, device=self.device)
            return

        ctx = torch.multiprocessing.get_context('spawn')
        pool = ctx.Pool(num_process)
        os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
        pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
        for i in range(num_process):
            sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
            pool.apply_async(
                self.inference_video,
                args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
                callback=lambda arg: pbar.update(1))
        pool.close()
        pool.join()

        # combine sub videos
        # prepare vidlist.txt
        with open(f'{args.output}/{args.video_name}_vidlist.txt', 'w') as f:
            for i in range(num_process):
                f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

        cmd = [
            args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', f'{args.output}/{args.video_name}_vidlist.txt', '-c',
            'copy', f'{video_save_path}'
        ]
        print(' '.join(cmd))
        subprocess.call(cmd)
        shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'))
        if osp.exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos')):
            shutil.rmtree(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'))
        os.remove(f'{args.output}/{args.video_name}_vidlist.txt')

    def inference_video(self, args, video_save_path, device=None, total_workers=1, worker_idx=0):
        # determin model
        self.determine_model(args)
        # determin model path
        model_path = self.determine_model_path(args)
        # use dni to control the denoise strength
        dni_weight = None
        if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
        # restorer
        upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=self.model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=not args.fp32,
            gpu_id=args.gpu_id,
            device=device,
        )
        if 'anime' in args.model_name and args.face_enhance:
            print('face_enhance is not supported in anime models, we turned this option off for you. '
                'if you insist on turning it on, please manually comment the relevant lines of code.')
            args.face_enhance = False

        if args.face_enhance:  # Use GFPGAN for face enhancement
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)  # TODO support custom device
        else:
            face_enhancer = None

        reader = Reader(args, total_workers, worker_idx)
        audio = reader.get_audio()
        height, width = reader.get_resolution()
        fps = reader.get_fps()
        writer = Writer(args, audio, height, width, video_save_path, fps)
        pbar = tqdm(total=len(reader), unit='frame', desc='inference')
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                writer.write_frame(output)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            else:
                torch.cuda.synchronize(device)
            pbar.update(1)
        reader.close()
        writer.close()

    def get_args(self):
            parser = argparse.ArgumentParser()
            parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
            parser.add_argument('-n', '--model_name', type=str, default='realesr-general-x4v3', help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x2plus | realesr-general-x4v3'))
            parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
            parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5, help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. Only used for the realesr-general-x4v3 model'))
            parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
            parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
            parser.add_argument('-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
            parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
            parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
            parser.add_argument('--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
            parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
            parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
            parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
            parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
            parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
            parser.add_argument('--extract_frame_first', action='store_true')
            parser.add_argument('--num_process_per_gpu', type=int, default=1)
            parser.add_argument('--alpha_upsampler', type=str, default='realesrgan', help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
            parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
            return parser.parse_args()

    def determine_model(self, args):
        args.model_name = args.model_name.split('.pth')[0]
        if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            self.netscale = 4
            self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.netscale = 2
            self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            self.netscale = 4
            self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            self.netscale = 4
            self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']

    def determine_model_path(self, args):
        if args.model_path is not None:
            model_path = args.model_path
        else:
            model_path = os.path.join('weights', args.model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in self.file_url: # model_path will be updated
                    model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        return model_path

class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error').run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']
        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]
            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()

class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')
        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret

def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path
