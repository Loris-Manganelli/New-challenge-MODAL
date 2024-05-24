import hydra
import os
import pandas as pd

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests




@hydra.main(config_path="configs/train", config_name="config")
def create_output(cfg):

    #Get test images
    test_dataset_path=cfg.dataset.test_path
    images_list = os.listdir(test_dataset_path)
    # filter out non-image files
    images_list = [image for image in images_list if image.endswith(".jpg")]
    
    idx=0
    image_name = images_list[idx]
    image_path = os.path.join(test_dataset_path, image_name)
    image = Image.open(image_path)

    llava_compute(image)
    submission = pd.DataFrame(columns=["id", "label"])
    


def llava_compute(image):
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
                                                          torch_dtype=torch.float16, 
                                                          low_cpu_mem_usage=True,
                                                          load_in_4bit=False, #INSTALLER pip install bitsandbytes
                                                          use_flash_attention_2=False) #INSTALLER https://github.com/Dao-AILab/flash-attention
    model.to("cuda:0")

    # prepare image and text prompt, using the appropriate prompt template
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))

create_output()