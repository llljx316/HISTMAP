
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration, AutoProcessor
from utils import iterative_all_files
import torch
from functools import partial
import pickle
import os
from pathlib import Path


processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b") 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
text_ls = []
text_dict = {}
# prompt= '''Describe a visual style for an image that captures [main subject, e.g., "a serene forest scene"] with a focus on [color palette, e.g., "cool, muted colors like soft blues and greens"]. This style should evoke [emotion or atmosphere, e.g., "tranquility and mystery"], using [lighting, e.g., "soft, diffused lighting that creates gentle shadows"]. Incorporate characteristics of [specific art style or era, if applicable, e.g., "Impressionist paintings, focusing on texture and light play"] for added visual interest. The style should also include any additional elements, e.g., "minimal details in the background to maintain focus on the main subject". The generated style is'''
prompt = "Question: Analyze the image and provide a concise, domain-specific description using terminology from the historical map domain. Answer one word. Answer:"
prompt2 = "Question: Analyze the image and provide a concise, domain-specific description using terminology from the historical map domain. Answer one sentence. Answer:"

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 


def blip_model(image_path, text_dict):
    raw_image = Image.open(image_path).convert('RGB')
    # conditional image captioning
    if prompt == "":
        inputs = processor(raw_image, return_tensors="pt").to(device)
    else:
        inputs = processor(raw_image, prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=100)
    # text_dict[image_path.stem] = processor.decode(out[0]).replace(prompt, "").strip()
    now_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if now_text == '':
        inputs = processor(raw_image, prompt2, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        now_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    text_dict[image_path.stem] = now_text

def cluster_sentences(text_ls):
    pass

# def semantic_filter()

if __name__ == '__main__':
    #generate sentence in a list
    # blip_model("result/segres/Bodleian Library/0ac39b91-cd26-4d05-a47c-5439aef2747d/11.png", "The pattern is")
    print(os.getcwd())
    

    
    sub_dataset_dir = Path('result/segres')

    # 列出第一层的内容
    first_level_dirs = [p.name for p in sub_dataset_dir.iterdir() if p.is_dir()]

    for dir in first_level_dirs:
        relative_dir = sub_dataset_dir/dir
        print(relative_dir)
        text_dict = {}
        process_image = partial(blip_model, text_dict = text_dict)
        iterative_all_files(relative_dir, process_image, suffix_filter=[".png"])
        #delete the last [SEP]
        text_dict.pop('final')
        for key, value in text_dict.items():
            text_dict[key] = value
        print(text_dict)
        with open(relative_dir/'text_res.pickle', 'wb') as f:
            pickle.dump(text_dict, f)


    # with open('./result/textres/text_embedding_style.pickle', 'wb') as pickle_file:
    #     pickle.dump(text_ls, pickle_file)
    #generate embedding
    # text_ls = [s[165:] for s in text_ls]
    # print(text_ls)
    

    #visualization


