import logging
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoCutManager:
    def __init__(self):
        pass

    def save_video(self, input_frames, output_path, fps=30):
        if isinstance(input_frames, list):
            # Convert list of PIL Images to a list of NumPy arrays
            frames = [np.array(frame) for frame in input_frames]
            # Stack along the time dimension to get a 4D array (t, h, w, c)
            input_frames = np.stack(frames, axis=0)
        assert input_frames.ndim == 4 and input_frames.shape[3] == 3, f"invalid shape: {input_frames.shape} (need t h w c)"
        if input_frames.dtype != np.uint8:
            input_frames = (input_frames * 255).astype(np.uint8)
        ImageSequenceClip(list(input_frames), fps=fps).write_videofile(output_path, codec='libx264')

    def concatenate_videos(self, video_A, video_B, video_merged):
        # ensure path
        video_A = str(video_A)
        video_B = str(video_B)
        video_merged = str(video_merged)
        # Load the two videos
        clip_A = VideoFileClip(video_A)
        clip_B = VideoFileClip(video_B)
        # get size
        width_A, height_A = clip_A.size
        width_B, height_B = clip_B.size
        logging.info(f"[width, height] of video A = [{width_A},{height_A}]")
        logging.info(f"[width, height] of video B = [{width_B},{height_B}]")
        # get fps
        fps_A = clip_A.fps
        fps_B = clip_B.fps
        logging.info(f"fps of video A = {fps_A}")
        logging.info(f"fps of video B = {fps_B}")
        # handle resolution
        if width_A != width_B or height_A != height_B:
            logging.warning(f"The input video clips have different sresolution. We force resize video B resolution to the video A resolution as width:{width_A}, height:{height_A} ")
            clip_B = clip_B.resize((width_A, height_A))
        # handle fps
        if fps_A > fps_B:
            clip_B = clip_B.set_fps(fps_A)
            logging.info(f"set video A fps to the video B")
        elif fps_B > fps_A:
            clip_A = clip_A.set_fps(fps_B)
            logging.info(f"set video B fps to the video A")
        # Concatenate the clips
        final_clip = concatenate_videoclips(clips=[clip_A, clip_B], method='compose')
        # Write the final video
        final_clip.write_videofile(video_merged, codec='libx264')

if __name__ == "__main__":
    videocut_manager = VideoCutManager()
    videocut_manager.concatenate_videos(
        video_A = "../videos/suv1.mp4",
        video_B = "../videos/suv2.mp4",
        video_merged = "../videos/merged.mp4"
    )
