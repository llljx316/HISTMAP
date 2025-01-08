import sys
sys.path.append("..")
import numpy as np
import faiss
import torch
from pipeline.utils import *
from pipeline.ImageBind import *

def create_index(image_embeddings, k, index_path):
    # Prepare your dataset
    data = np.array(image_embeddings.cpu(), dtype=np.float32)  # Replace [...] with your data
    d = data.shape[1]  # Dimensionality
    # nlist = 100  # Number of clusters
    nlist = int(k * np.sqrt(data.shape[0]))  # Results in nlist = 4000


    # Initialize the quantizer and index
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train the index
    index.train(data)

    # Add data to the index
    index.add(data)
    faiss.write_index(index, index_path)
    return index


class IVFIndex:
    def init_model(self):
        return ImageBindPipeline()

    def __init__(self, load_index=True) -> None:
        self.device = DEVICE
        self.model = self.init_model()
        self.info_dict_full = load_info_dict('result/info_dict_new_cpu.pickle')
        self.info_dict = self.info_dict_full
        #generate the matrix
        self.image_embeddings = load_image_embeddings(self.info_dict)
        self.id_info_list = load_id_info_list(self.info_dict)
        self.text_embeddings = load_text_embeddings(self.info_dict)
        self.k = 1#20
        self.FULL = not load_index

        #division by datasets
        info_dict_datasets_path = Path("./modelApi/info_dict_new_datasets.pickle")
        if info_dict_datasets_path.exists():
            print("datasets division file exsists!")
            self.info_dict_datasets = load_info_dict(str(info_dict_datasets_path)) 
        else:
            #division
            print("datasets division file not exsists! start generate!")
            self.info_dict_datasets = {}
            for p in self.info_dict.keys():
                fpath = Path(p)
                dataset_name = fpath.parts[2]
                if dataset_name not in self.info_dict_datasets:
                    self.info_dict_datasets[dataset_name] = {}
                self.info_dict_datasets[dataset_name][p] = self.info_dict[p]
            save_info_dict(self.info_dict_datasets, str(info_dict_datasets_path)) 
            print("finish generation")
            

        if load_index:
            index_path = "./modelApi/trained_index.faiss"
            text_index_path = "./modelApi/trained_index_text.faiss" 
            if Path(index_path).exists() and Path(text_index_path).exists():
                self.index = faiss.read_index(index_path)
                self.text_index = faiss.read_index(text_index_path)
            else:
                # # Prepare your dataset
                # data = np.array(self.image_embeddings.cpu(), dtype=np.float32)  # Replace [...] with your data
                # d = data.shape[1]  # Dimensionality
                # # nlist = 100  # Number of clusters
                # nlist = int(self.k * np.sqrt(data.shape[0]))  # Results in nlist = 4000


                # # Initialize the quantizer and index
                # quantizer = faiss.IndexFlatL2(d)
                # self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

                # # Train the index
                # self.index.train(data)

                # # Add data to the index
                # self.index.add(data)
                # faiss.write_index(self.index, index_path)
                self.index = create_index(self.image_embeddings, self.k, index_path)
                self.text_index = create_index(self.text_embeddings, self.k, text_index_path)
    
    def filter_by_dataset(self, dataset_names):
        self.info_dict = {}
        for dataset in dataset_names:
            self.info_dict.update(self.info_dict_datasets[dataset])
        #generate the matrix
        self.image_embeddings = load_image_embeddings(self.info_dict)
        self.id_info_list = load_id_info_list(self.info_dict)
        self.text_embeddings = load_text_embeddings(self.info_dict)
        self.FULL = True if len(dataset_names)!=11 else False

    def query(self, embedding, num, on):
        if on == 'image':
            D,I = self.index.search(np.array(embedding.detach().cpu()),num)
        else:
            D,I = self.text_index.search(np.array(embedding.detach().cpu()),num)
        index = np.array(list(filter(lambda x: x!=-1,I[0])))
        return np.array(self.id_info_list)[index]
    
    def full_query(self, embedding, num, on='image'):
        similarity =(self.image_embeddings.to(self.device) if on == 'image' else self.text_embeddings.to(self.device)) @ embedding.T 
        try:
            _,i = torch.topk(similarity.flatten(), num)
            i = i.cpu()
        except:
            i=list(range(similarity.shape[0]))

        return np.array(self.id_info_list)[i]


    def full_query_statistic_num(self, text, threshold, on='image'):
        text_embedding = self.model.generate_word_embedding([text])# must be list
        text_embedding = text_embedding[ModalityType.TEXT]
        arr =(self.image_embeddings.to(self.device) if on == 'image' else self.text_embeddings.to(self.device)) @ text_embedding.T 
        # arr_min = torch.min(arr)
        arr_max = torch.max(arr)
        normalized_arr = arr/arr_max
        # _,i = torch.topk(similarity.flatten(), num)
        # i = i.cpu()
        return torch.sum(normalized_arr > threshold).item()



    def query_text(self, text, num=10, on = 'image', full = False):
        full = self.FULL
        text_embedding = self.model.generate_word_embedding([text])# must be list
        # return self.full_query(text_embedding[ModalityType.TEXT],num, on)
        return (full and self.full_query or self.query)(text_embedding[ModalityType.TEXT],num, on)

    def query_image_path(self, img_path, num=10, on = 'image', full=False):
        full = self.FULL
        image_embedding = self.model.generate_image_embeddings([img_path])
        return (full and self.full_query or self.query)(image_embedding[ModalityType.VISION],num, on)

    def query_id(self, id, num=10, on = 'image'):
        embedding = self.id_info_list[id]['image_embedding'].to(self.device)
        return self.full_query(embedding,num, on)


    def generateUMAPJson(self):
        pass



