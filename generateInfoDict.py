from pathlib import Path
import pickle
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np
from tqdm import tqdm

#! after debug
# model = imagebind_model.imagebind_huge(pretrained=True)
# model.eval()
# model.to(device)


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def save_pickle(obj, save_dir):
    with open(save_dir, 'wb') as f:
        pickle.dump(obj,f)

def load_pickle(path):
    with open('result/info_dict.pickle', 'rb') as f:
        lf= pickle.load(f)
    return lf
def first_info_dict():
    #get image and text
    res_dir = Path('result/segres')
    #use dict to generate the root is the id
    #!!!!changed need to modify
    two_level_folder = [two_folder for two_folder in res_dir.rglob('*') if two_folder.is_dir() and two_folder.parent.parent==res_dir and (two_folder/'text_res.pickle').exists()]
    print(two_level_folder)
    info_dict = {}

    for folder in tqdm(two_level_folder):
        #load text
        with open(folder/ 'text_res.pickle', 'rb') as f:
            text_dict = pickle.load(f)
        #load iou
        with open(folder/ 'anns.pickle', 'rb') as f:
            iou_dict = pickle.load(f)
        
        #match it with path
        for file in folder.glob('*.png'):
            if file.name == 'final.png':
                continue
            info_dict[str(file)] = {'text': text_dict[file.stem], 'iou': iou_dict[int(file.stem)]}


    save_pickle(info_dict, 'result/info_dict_new.pickle')
    print(info_dict)
    return info_dict



def add_embeddings(info_dict):
    res_dir = Path('result/segres')
    # image_paths = list(res_dir.rglob('*png')) #! not right need change
    # image_paths = [file_path for file_path in image_paths if file_path.name != 'final.png']

    # image_paths = list(res_dir.rglob('*.png'))
    # image_paths = [file for file in image_paths if (file.parent/'text_res.pickle').exists() and file.name != 'final.png']
    image_paths = list(info_dict.keys())
    text_list = [info_dict[str(pic_path)]['text'] for pic_path in image_paths]
    # instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    batch_size = 32
    #shuffle
    indices = np.random.permutation(len(image_paths))
    image_paths = np.array(image_paths)[indices]
    text_list = np.array(text_list)[indices]
    image_paths_batches = np.array_split(image_paths, len(image_paths) // batch_size)
    text_ls_batches = np.array_split(text_list, len(text_list) // batch_size)
    #mini batch evaluate
    for image_path_batch, text_ls_batch in zip(tqdm(image_paths_batches), text_ls_batches):
        # Load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_ls_batch, device),
            ModalityType.VISION: data.load_and_transform_vision_data(image_path_batch, device),
            # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        #save to info_dict
        for i,path in enumerate(image_path_batch):
            print(path)
            info_dict[str(path)]['image_embedding'] = embeddings[ModalityType.VISION][i]
            info_dict[str(path)]['text_embedding'] = embeddings[ModalityType.TEXT][i]

    save_pickle(info_dict, 'result/info_dict_new.pickle')
    print(info_dict)


def info_dict_add_id(info_dict):
    i=0
    for key in info_dict:
        info_dict[key]['id']=i
        i+=1

    save_pickle(info_dict, 'result/info_dict_new.pickle')



if __name__=='__main__':
    info_dict = first_info_dict()
    # info_dict = load_pickle('result/info_dict_new.pickle') 
    add_embeddings(info_dict)
    info_dict_add_id(info_dict)