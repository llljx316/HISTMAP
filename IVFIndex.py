
import numpy as np
import faiss
import torch
from pipeline.utils import *
from pipeline.ImageBind import *


class IVFIndex:
    def init_model(self):
        return ImageBindPipeline()

    def __init__(self) -> None:
        self.model = self.init_model()
        self.info_dict = load_info_dict()
        #generate the matrix
        self.image_embeddings = load_image_embeddings(self.info_dict)
        self.id_info_list = load_id_info_list(self.info_dict)
        self.text_embeddings = load_text_embeddings(self.info_dict)
        self.k = 5
        # Prepare your dataset
        data = np.array(self.image_embeddings.cpu(), dtype=np.float32)  # Replace [...] with your data
        d = data.shape[1]  # Dimensionality
        # nlist = 100  # Number of clusters
        nlist = int(self.k * np.sqrt(data.shape[0]))  # Results in nlist = 4000


        # Initialize the quantizer and index
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train the index
        self.index.train(data)

        # Add data to the index
        self.index.add(data)

    def query(self, embedding):
        D,I = self.index.search(np.array(embedding.detach().cpu()),10)
        return np.array(self.id_info_list)[I[0]]

    def query_text(self, text):
        text_embedding = self.model.generate_word_embedding(text)
        return self.query(text_embedding[ModalityType.TEXT])

    def query_image_path(self, img_path):
        image_embedding = self.model.generate_image_embeddings(img_path)
        return self.query(image_embedding[ModalityType.VISION])



