import torch
import umap
import sys
import os
# sys.path.append("..")
# os.chdir('../')
from pathlib import Path
from pipeline.utils import *
from collections import Counter
import re
import pipeline.wizmap_new as wizmap
from sklearn.manifold import TSNE

METHOD = "tsne"
#common words
common_words_ls=["The","image","shows","a","large,","with","and","very","on","top","shaped","object","large","base","small","map","of","the","shown","in","branch","hole","center","goat","cylindrical","base","field","foreground","background","it"]


def static_for_word_count(id_info_dict):
    text = [id_info_dict[id]['text'] for id in range(len(id_info_dict))]
    # 对一个字符串数组进行词频统计的代码示例
    # 将数组中的每个字符串分割成单词
    words = []
    for string in text:
        words.extend(string.split())
    
    # 使用Counter统计词频
    word_count = Counter(words)
    return word_count

def umap_attr(image_emb, text_emb):
    embeddings = torch.cat((image_emb, text_emb), dim=0).cpu()
    umap_instance = umap.UMAP(n_components=2, random_state=42)
    low_dim_embeddings = umap_instance.fit_transform(embeddings)
    image_embeddings_2d = low_dim_embeddings[:len(image_emb)]
    text_embeddings_2d = low_dim_embeddings[len(image_emb):]
    return image_embeddings_2d,text_embeddings_2d  

def tsne_attr(image_emb, text_emb):
    embeddings = torch.cat((image_emb, text_emb), dim=0).cpu()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
# X_embedded = tsne.fit_transform(X)
    low_dim_embeddings = tsne.fit_transform(embeddings)
    image_embeddings_2d = low_dim_embeddings[:len(image_emb)]
    text_embeddings_2d = low_dim_embeddings[len(image_emb):]
    return image_embeddings_2d,text_embeddings_2d  

# def generate_json(id_info_dict, image_emb, text_emb):
#     image_json = [[emb[0], emb[1],id_info_dict[i]['path'], "", 1] for i,emb in enumerate(image_emb)]
#     text_json = [[emb[0], emb[1],id_info_dict[i]['text'], "", 0] for i,emb in enumerate(text_emb)]
#     json_array = image_json + text_json
#     # 将数据写入到文件
#     with open("umap.ndjson", "w", encoding="utf-8") as file:
#         for row in json_array:
#             # 将每行格式化成所需形式
#             formatted_row = "[" + ", ".join(f'"{item}"' if isinstance(item, str) else str(item) for item in row) + "]\n"
#             file.write(formatted_row)

#     print("数据已写入文件")

def text_spe_gen(strings):
    for word in common_words_ls:
        pattern = rf"\b{word}\b"  # \b 确保是完整的单词
        strings = [re.sub(pattern, "", s, flags=re.IGNORECASE).strip() for s in strings]  # 不区分大小写
    return strings

def generate_by_wizmap(id_info_dict, image_emb, text_emb):
    # emb_2d = torch.stack((text_emb, image_emb))

    # reverse image and text order
    # emb_2d = np.concatenate((image_emb,text_emb), 0)
    # xs = emb_2d.T[0].astype(float)
    # ys  = emb_2d.T[1].astype(float)
    # label = [0]*image_emb.shape[0] + [1]*text_emb.shape[0]
    # text = [id_info_dict[id]['path'] for id in range(image_emb.shape[0])] + [id_info_dict[id]['text'] for id in range(image_emb.shape[0])] 
    # texts_spe = [id_info_dict[id]['text'] for id in range(image_emb.shape[0])]*2
    # texts_spe = text_spe_gen(texts_spe)

    # #generate grid

    # grid_dict = wizmap.generate_grid_dict(
    #     xs, 
    #     ys, 
    #     text, 
    #     "historical maps elements embedding", 
    #     labels = label, 
    #     group_names=["image", "text"], 
    #     image_label = 0, 
    #     image_url_prefix="http://127.0.0.1:8002/",
    #     texts_spe=texts_spe    
    # )
    # data_list = wizmap.generate_data_list(xs, ys, text, labels=label)
    # wizmap.save_json_files(data_list, grid_dict, "./pipeline")
    # print('写入完成')



    emb_2d = np.concatenate((text_emb, image_emb), 0)
    xs = emb_2d.T[0].astype(float)
    ys  = emb_2d.T[1].astype(float)
    label = [0]*text_emb.shape[0] + [1]*image_emb.shape[0]
    text = [id_info_dict[id]['text'] for id in range(image_emb.shape[0])] + [id_info_dict[id]['path'] for id in range(image_emb.shape[0])]
    texts_spe = [id_info_dict[id]['text'] for id in range(image_emb.shape[0])]*2
    texts_spe = text_spe_gen(texts_spe)

    #generate grid

    grid_dict = wizmap.generate_grid_dict(
        xs, 
        ys, 
        text, 
        "historical maps elements embedding", 
        labels = label, 
        group_names=["text", "image"], 
        image_label = 1, 
        image_url_prefix="http://127.0.0.1:8002/",
        texts_spe=texts_spe    
    )
    data_list = wizmap.generate_data_list(xs, ys, text, labels=label)
    wizmap.save_json_files(data_list, grid_dict, "./pipeline")
    print('写入完成')


# info_dict = load_info_dict('result/info_dict_new_cpu.pickle')
info_dict = load_info_dict()
id_info_dict = load_id_info_list(info_dict)
umap_pickle_path = Path(f'result/{METHOD}.pickle')
if umap_pickle_path.exists():
    print("load!")
    with open(str(umap_pickle_path), 'rb') as f:
        img_emb_2d, text_emb_2d = pickle.load(f)
else:
    img_emb = load_image_embeddings(info_dict)
    text_emb = load_text_embeddings(info_dict)
    if METHOD == "tsne":
        img_emb_2d, text_emb_2d = tsne_attr(img_emb, text_emb)
    else:
        img_emb_2d, text_emb_2d = umap_attr(img_emb, text_emb)
    with open(str(umap_pickle_path), 'wb') as f:
        pickle.dump((img_emb_2d, text_emb_2d),f)
# generate_json(id_info_dict, img_emb_2d, text_emb_2d)
generate_by_wizmap(id_info_dict, img_emb_2d, text_emb_2d)
result = static_for_word_count(id_info_dict)
# 输出结果
with open("word_frequency_log.txt", "w") as log_file:
    for word, count in result.items():
        log_file.write(f"{word}: {count}\n")


with open("word_list_log.txt", "w") as log_file:
    for word, count in result.items():
        log_file.write(f"\"{word}\",")
