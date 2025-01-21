import requests
import soundfile as sf
from kokoro_onnx import Kokoro
from tqdm import tqdm 
from huggingface_hub import snapshot_download
from utils.helper import check_and_make_folder

# define
REPO_ID = "hexgrad/Kokoro-82M"
MODEL_CACHE = "kokoro-cache"
ONNX_NAME = "kokoro-v0_19.onnx"
JSON_NAME = "voices.json"
JSON_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/" + JSON_NAME
ONNX_PATH = MODEL_CACHE + "/" + ONNX_NAME
JSON_PATH = MODEL_CACHE + "/" + JSON_NAME
OUTPUT_PATH = "output_kokoro"

def download_models():
    print(f"start download onnx file")
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[ONNX_NAME],
        local_dir=MODEL_CACHE,
    )
    print(f"start download json file")    
    response = requests.get(JSON_URL, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(JSON_PATH, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong during download")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# check output folder
check_and_make_folder(OUTPUT_PATH)

# only run in first time
if False: 
    download_models()

# set input text
input_text = "The sun dipped below the horizon, painting the sky in hues of orange and pink. \
    The gentle breeze carried the scent of blooming flowers, and the distant sound of waves crashing against the shore added a soothing rhythm to the evening. \
    It was moments like these that reminded her of the beauty in simplicity, and how nature had a way of grounding the soul"

print(f"start generate audio")
kokoro = Kokoro(ONNX_PATH, JSON_PATH)
samples, sample_rate = kokoro.create(
    input_text, 
    voice="af",
    lang="en-us",
    speed=1.0, 
)

print(f"start save audio")
sf.write(OUTPUT_PATH + "/audio.wav", samples, sample_rate)
print(f"Finish create .wav file")
