import os
import time
import cv2
import torch
import argparse
import numpy as np
import _thread
import skvideo.io
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.nn import functional as F
from queue import Queue, Empty
from src.RIFE.pytorch_msssim import ssim_matlab
from src.RIFE.train_log.RIFE_HDv3 import Model

class RIFEManager:
    def __init__(self, device : torch.device, dtype : torch.dtype):
        #torch.set_grad_enabled(False)
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.model = None

    def setup(self, args):        
        print("Loaded v3.x HD model.")
        self.model = Model(mps=True)
        self.model.load_model(args.modelDir, -1)
        self.model.eval()
        self.model.device()

    def process_img(self, args):
        assert (not args.img is None)
        print("Start a image process!")
        print(f"Set args to: '{args}'")

        if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
            img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
        else:
            img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if args.ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if args.ratio <= img0_ratio + args.rthreshold / 2:
                middle = img0
            elif args.ratio >= img1_ratio - args.rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(args.rmaxcycles):
                    middle = self.model.inference(tmp_img0, tmp_img1)
                    middle_ratio = ( img0_ratio + img1_ratio ) / 2
                    if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                        break
                    if args.ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for i in range(args.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        if not os.path.exists(args.output):
            os.mkdir(args.output)
        for i in range(len(img_list)):
            if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
                cv2.imwrite(args.output+'/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            else:
                cv2.imwrite(args.output+'/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

    def process_video(self, args):
        assert (not args.video is None)
        args.img = None # have to set 'img' to 'None' first
        
        print("Start a video process!")
        print(f"Set args to: '{args}'")

        def make_inference(I0, I1, n):
            middle = self.model.inference(I0, I1, args.scale)
            if n == 1:
                return [middle]
            first_half = make_inference(I0, middle, n=n//2)
            second_half = make_inference(middle, I1, n=n//2)
            if n%2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]
        
        if args.skip:
            print("skip flag is abandoned, please refer to issue #207.")

        if args.UHD and args.scale==1.0:
            args.scale = 0.5
        assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

        if not args.img is None:
            args.png = True
        
        if not args.video is None:
            videoCapture = cv2.VideoCapture(args.video)
            fps = videoCapture.get(cv2.CAP_PROP_FPS)
            tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
            videoCapture.release()
            fpsNotAssigned = True
            if args.fps is None:
                args.fps = fps * (2 ** args.exp)
            videogen = skvideo.io.vreader(args.video)
            lastframe = next(videogen)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_path_wo_ext, ext = os.path.splitext(args.video)
            print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
            if args.png == False and fpsNotAssigned == True:
                print("The audio will be merged after interpolation process")
            else:
                print("Will not merge audio because using png or fps flag!")
        else:
            videogen = []
            for f in os.listdir(args.img):
                if 'png' in f:
                    videogen.append(f)
            tot_frame = len(videogen)
            videogen.sort(key=lambda x: int(x[:-4]))
            lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            videogen = videogen[1:]

        h, w, _ = lastframe.shape
        vid_out_name = None
        vid_out = None
        if args.png:
            if not os.path.exists('vid_out'):
                os.mkdir('vid_out')
        else:
            if args.output is not None:
                vid_out_name = args.output
            else:
                vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
            vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

        if args.montage:
            left = w // 4
            w = w // 2
        tmp = max(32, int(32 / args.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        pbar = tqdm(total=tot_frame)
        if args.montage:
            lastframe = lastframe[:, left: left + w]
        write_buffer = Queue(maxsize=500)
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, vid_out))

        I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, padding, args)
        temp = None  # save lastframe when processing static frame

        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_buffer.get()
            if frame is None:
                break
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1, padding, args)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            break_flag = False
            if ssim > 0.996:
                frame = read_buffer.get()  # read a new frame
                if frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = frame
                I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                I1 = pad_image(I1, padding, args)
                I1 = self.model.inference(I0, I1, args.scale)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

            if ssim < 0.2:
                output = []
                for i in range((2 ** args.exp) - 1):
                    output.append(I0)
                '''
                output = []
                step = 1 / (2 ** args.exp)
                alpha = 0
                for i in range((2 ** args.exp) - 1):
                    alpha += step
                    beta = 1-alpha
                    output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
                '''
            else:
                output = make_inference(I0, I1, 2 ** args.exp - 1) if args.exp else []

            if args.montage:
                write_buffer.put(np.concatenate((lastframe, lastframe), 1))
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
            else:
                write_buffer.put(lastframe)
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    write_buffer.put(mid[:h, :w])
            pbar.update(1)
            lastframe = frame
            if break_flag:
                break

        if args.montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        else:
            write_buffer.put(lastframe)

        write_buffer.put(None)

        while (not write_buffer.empty()):
            time.sleep(0.1)
        pbar.close()
        if not vid_out is None:
            vid_out.release()

        # move audio to new video file if appropriate
        if args.png == False and fpsNotAssigned == True and not args.video is None:
            try:
                transferAudio(args.video, vid_out_name)
            except:
                print("Audio transfer failed. Interpolated video will have no audio")
                targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
                os.rename(targetNoAudio, vid_out_name)

    def get_args(self):
        parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
        parser.add_argument('--video', dest='video', type=str, default=None)
        parser.add_argument('--output', dest='output', type=str, default=None)
        parser.add_argument('--img', dest='img', type=str, default=None, nargs=2)
        parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
        parser.add_argument('--model', dest='modelDir', type=str, default='src/rife/train_log', help='directory with trained model files')
        parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
        parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
        parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
        parser.add_argument('--fps', dest='fps', type=int, default=None)
        parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
        parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
        parser.add_argument('--exp', dest='exp', type=int, default=1)
        parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
        parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
        parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
        return parser.parse_args()

def transferAudio(sourceVideo, targetVideo):
    import shutil
    #import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"
    # split audio from original video file and store in "temp" directory
    if True:
        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))
    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)
    # remove temp directory
    shutil.rmtree("temp")

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
             if not user_args.img is None:
                  frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
             if user_args.montage:
                  frame = frame[:, left: left + w]
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def pad_image(img, padding, args):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

def clear_write_buffer(user_args, write_buffer, vid_out):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])
