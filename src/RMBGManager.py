import os
import gc
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from utils.helper import check_and_make_folder

REPO_ID = "briaai/RMBG-2.0"
MODEL_CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--briaai--RMBG-2.0/")

def crop_bottom_transparent(layer_img):
    """
    去除图层底部完全透明的像素。
    """
    # 获取图像数据
    datas = layer_img.getdata()
    width, height = layer_img.size

    # 找到底部第一个非透明像素的y坐标
    bottom_y = height
    for y in range(height-1, -1, -1):
        row = layer_img.crop((0, y, width, y+1))
        if row.getbbox():  # 如果这一行有非透明像素
            bottom_y = y + 1
            break

    # 裁剪图像
    cropped_img = layer_img.crop((0, 0, width, bottom_y))
    return cropped_img

def process_layer(layer_img, bg_width, bg_height, align="bottom"):
    """
    处理单个图层：
    - 等比例缩放到与背景图片同宽
    - 高度不足时，底部填充透明背景
    - 高度超出时，底部对齐并裁剪多余部分
    """
    # 确保图层具有 alpha 通道
    layer_img = layer_img.convert("RGBA")
    
    # 获取原始尺寸
    orig_width, orig_height = layer_img.size
    
    # 计算缩放比例
    scale_ratio = bg_width / orig_width
    new_width = bg_width
    new_height = int(orig_height * scale_ratio)
    
    # 缩放图层
    resized_layer = layer_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    if new_height < bg_height:
         # 高度不足，填充透明背景
        new_layer = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
        if align == 'bottom':
            paste_position = (0, bg_height - new_height)
        elif align == 'top':
            paste_position = (0, 0)
        elif align == 'center':
            paste_position = (0, (bg_height - new_height) // 2)
        else:
            raise ValueError("align 参数必须是 'top', 'center' 或 'bottom'")
        new_layer.paste(resized_layer, paste_position, resized_layer)
    else:
       # 高度超出，裁剪多余部分
        excess_height = new_height - bg_height
        if align == 'bottom':
            crop_box = (0, excess_height, bg_width, new_height)
        elif align == 'top':
            crop_box = (0, 0, bg_width, bg_height)
        elif align == 'center':
            crop_box = (0, excess_height // 2, bg_width, excess_height // 2 + bg_height)
        else:
            raise ValueError("align 参数必须是 'top', 'center' 或 'bottom'")
        cropped_resized_layer = resized_layer.crop(crop_box)
        new_layer = cropped_resized_layer
    
    return new_layer

class RMBGManager:
    def __init__(self, device : torch.device, dtype : torch.dtype):
        self.device : torch.device = device
        self.dtype: torch.dtype = dtype
        self.output_path : str = "output_rmbg"
        self.pipe = None

    def cleanup(self):
        print("Run cleanup")
        gc.collect()
        torch.mps.empty_cache()

    def setup(self) -> None:
        print("chekc output folder")
        check_and_make_folder(self.output_path)

        print("chekc models & init pipeline")
        self.pipe = AutoModelForImageSegmentation.from_pretrained(
            REPO_ID,
            cache_dir=MODEL_CACHE,
            trust_remote_code=True
        )
        torch.set_float32_matmul_precision(['high', 'highest'][0])

        self.pipe.to(self.device)
        self.pipe.eval()

        # image processor
        image_size = (1024, 1024)
        self.image_processor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def generate(self, 
                 input_image,
                 background_image=None
        ) -> None:
        print("start remove background image")
        output_image_path = self.output_path + "/final_image.png"

        image = Image.open(str(input_image)).convert('RGB')
        input_images = self.image_processor(image).unsqueeze(0).to(self.device)

        # image background removal
        with torch.no_grad():
            preds = self.pipe(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)

        if background_image is None:
            print("save output image")
            image.save(output_image_path)
            return
        
        print("start add background image")
        layer_img = crop_bottom_transparent(image)
        bg = Image.open(background_image).convert("RGBA")
        bg_width, bg_height = bg.size
        processe_middle_layer = process_layer(layer_img, bg_width, bg_height)
        combined = Image.alpha_composite(bg, processe_middle_layer)

        print("save output image")
        combined.save(output_image_path)
