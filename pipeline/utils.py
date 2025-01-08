#Environment
from pathlib import Path
# from imagebind import data
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap

DEVICE = 'cuda:1'

def load_info_dict(path='result/info_dict_new_cpu.pickle'):
    with open(path, 'rb') as f:
        info_dict = pickle.load(f)
        # info_dict = torch.load(f, map_location=torch.device('cpu'))
    return info_dict

def save_info_dict(obj,path):
    with open(path, 'wb') as f:
        info_dict = pickle.dump(obj, f)
        # info_dict = torch.load(f, map_location=torch.device('cpu'))

def load_id_info_list(info_dict):
    id_info_dict = {info['id']: {**info, 'path':key} for key,info in info_dict.items()}
    id_info_list = [id_info_dict[key] for key in sorted(id_info_dict)]
    return id_info_list

def info_dict_to_cpu(path='result/info_dict_new.pickle'):
    with open(path, 'rb') as f:
        info_dict = pickle.load(f)
    for key in info_dict.keys():
        info_dict[key]['image_embedding']=info_dict[key]['image_embedding'].cpu()
        info_dict[key]['text_embedding']=info_dict[key]['text_embedding'].cpu()
    with open("result/info_dict_new_cpu.pickle", 'wb') as f:
        pickle.dump(info_dict,f)
    



def load_image_embeddings(info_dict):
    id_info_list = load_id_info_list(info_dict)
    image_embeddings = torch.stack( [info['image_embedding'] for info in id_info_list])
    return image_embeddings

def load_text_embeddings(info_dict):
    id_info_list = load_id_info_list(info_dict)
    text_embeddings = torch.stack( [info['text_embedding'] for info in id_info_list])
    return text_embeddings

def show_results(img_paths, row_len = 10, figsize=20):
    # 创建一个 n 行 10 列的图形布局
    # row_len = 10
    n = int((len(img_paths)-1)/row_len +1)
    fig, axes = plt.subplots(n, row_len, figsize=(figsize, figsize)) #10置合适的宽高

    # 逐个读取图片并显示
    print(axes)
    for i, img_path in enumerate(img_paths):
        if n == 1:
            now_axes = axes[i%row_len]
        else:
            now_axes = axes[int(i/row_len)][i%row_len]
        img = mpimg.imread(img_path)  # 读取图片
        now_axes.imshow(img)           # 显示图片
        now_axes.axis('off')
    # plt.axis('off')  #
    plt.tight_layout()
    plt.show()



def show_results_with_text(img_paths, info_dict):
    row_len = 10
    n = int((len(img_paths)-1)/row_len +1)
    fig, axes = plt.subplots(n, row_len, figsize=(20, 20))  # 设置合适的宽高

    # 逐个读取图片并显示
    for i, img_path in enumerate(img_paths):
        if n == 1:
            now_axes = axes[i%row_len]
        else:
            now_axes = axes[int(i/row_len)][i%row_len]
        img = mpimg.imread(img_path)  # 读取图片
        now_axes.imshow(img)           # 显示图片
        #similar image text
        # 使用 textwrap.wrap 将文字分行
        wrapped_caption = "\n".join(textwrap.wrap(f"{str(img_path)}: {info_dict[img_path]['text']}", 25))

        # 设置标题，限制字体大小
        now_axes.set_title(wrapped_caption, fontsize=10)  # 设置小字体大小
        now_axes.axis('off')           # 关闭坐标轴
    plt.axis('off')
    plt.tight_layout()
    plt.show()


