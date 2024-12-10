from IVFIndex import *
from fastapi import FastAPI
from pydantic import BaseModel
import os
os.chdir('../')

faiss = IVFIndex()
app = FastAPI()

class inputdata(BaseModel):
    text: str

@app.get("/text_predict")
def predict_text(text_input: str, num: int = 10):
    result = faiss.query_text(text_input, num)
    ret_res = [{"text": d['text'], 'path': d['path']}for d in result] 
    return ret_res


@app.get("/image_predict")
def predict_text(image_path: str, num: int = 10):
    image_path = Path(__file__).parents[1] / image_path
    result = faiss.query_image_path(image_path, num)
    ret_res = [{"text": d['text'], 'path': d['path']}for d in result] 
    return ret_res
