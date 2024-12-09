from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np

class ImageBindPipeline:
    def __init__(self):

        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

        # instantiate model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    def generate_word_embedding(self, text_input: np.array):
        sentences_embeddings = self.model({ModalityType.TEXT:data.load_and_transform_text(text_input, self.device)})
        return sentences_embeddings
    
    def generate_image_embeddings(self, image_paths: np.array):
        return self.model({ModalityType.VISION:data.load_and_transform_vision_data(image_paths, self.device)})
    
    def generate_embeddings(self, input):
        return self.model(input)
    
    

