import os
#os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
#os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg"
import numpy as np
import stat
import subprocess
import torch
import platform
from PIL import Image

def set_device():
    print('Pytorch version', torch.__version__)
    if torch.backends.mps.is_available():
        print("Set device to 'mps'")
        device = torch.device('mps')
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_VISUALIZE_ALLOCATIONS"] = "1"
        os.environ["PYTORCH_MPS_TENSOR_CORE_ENABLED"] = "1"
        os.environ["ACCELERATE_USE_MPS_DEVICE"] = "1"
        #os.environ["PYTORCH_MPS_PINNED_MAX_MEMORY_RATIO"] = "0.0"
        #os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # 最好不用, 用了也超级慢
    elif torch.cuda.is_available():
        print("Set device to 'cuda'")
        device = torch.device('cuda', 0)
    else:
        print("Set device to 'cpu'")
        device = torch.device('cpu')
    return device

def check_numpy_version():
    np_version = [int(i) for i in np.__version__.split('.')]
    print("numpy version: ", np_version)
    if np_version[0] == 2 or (np_version[0] == 1 and np_version[1] >= 20):
        np.float = float
        np.int = int

def show_img(file_path : str, title : str) -> None:
    if platform.system() == "Darwin":
        subprocess.run(["osascript", "-e", 'tell application "Preview" to quit'], check=False) # close last preview window
        subprocess.run(["open", file_path]) # show current preview window
    else:
        img = Image.open(file_path)
        img.show(title)

def calc_time_consumption(start_time, end_time) -> None:
    if end_time == 0 and start_time == 0:
        print("Warning: both 'end time' and 'start time' are 0.0. no time calculation can be performed.")
        return
    elapsed_time = (end_time - start_time) / 60.0
    print(f"Time taken: {elapsed_time:.2f} minutes totally")

def remove_files_except_with_suffix(folder_path : str, suffix : str) -> None:
    for file in os.listdir(folder_path):
        if not file.endswith(suffix):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def remove_all_files(folder_path : str) -> None:
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def check_and_make_folder(output_path : str) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

def check_and_init_folder(folder_path : str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if platform.system() == "Darwin":
            os.chmod(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO) # 777 permission
    else:
        remove_all_files(folder_path)

def get_new_folder_name_with_index(folder_name, index):
    return f"{folder_name}/img_{index}"

def get_new_object_name_with_index(path : str, index : int) -> str:
    return f"{ path.split('.')[0] }_{ index }.{ path.split('.')[1] }"

def find_single_file_with_suffix(folder_path : str, suffix : str) -> str:
    for file in os.listdir(folder_path):
        if file.endswith(suffix):
            return os.path.join(folder_path, file)
    return None
