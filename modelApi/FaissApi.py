from IVFIndex import *
from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
os.chdir('../')
# sys.path.append("..")

faiss = IVFIndex()
app = FastAPI()
datasetLs = ["Bodleian Library", "Bibliotheque Nationale de France", "John Carter Brown Library"]

class inputdata(BaseModel):
    text: str

@app.get("/text_predict")
def predict_text(text_input: str, num: int = 10, on:str = 'image'):
    result = faiss.query_text(text_input, num, on)
    # result = filter(lambda x: x!=-1, result)
    ret_res = [{"text": d['text'], 'path': d['path']}for d in result] 
    return ret_res


@app.get("/image_predict")
def predict_text(image_path: str, num: int = 10, on:str = 'image'):
    image_path = Path(__file__).parents[1] / image_path
    result = faiss.query_image_path(image_path, num, on)
    # result = filter(lambda x: x!=-1, result)
    ret_res = [{"text": d['text'], 'path': d['path']}for d in result] 
    return ret_res
 
@app.get("/id_predict")
def predict_id(id: int, num:int = 10, on:str = 'image'):
    result = faiss.query_id(id, num, on)
    ret_res = [{"text": d['text'], 'path': d['path']}for d in result] 
    return ret_res

@app.get("/distance_num")
def distance_num(text_input:str, threshold:float, on:str = 'image'):
    return {"num": faiss.full_query_statistic_num(text_input, threshold, on)}

from pydantic import BaseModel
from typing import List

class DatasetRequest(BaseModel):
    dataset_name: List[str]

@app.post("/filter_dataset")
async def filter_dataset(request:  DatasetRequest):
    print(request)
    dataset_names = request.dataset_name
    faiss.filter_by_dataset(dataset_names)
    return {"status": "success", "message": "Operation completed successfully"}

