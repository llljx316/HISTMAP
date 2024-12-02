from IVFIndex import *
from fastapi import FastAPI
from pydantic import BaseModel

faiss = IVFIndex()
app = FastAPI()

class inputdata(BaseModel):
    text: str

@app.post("/text_predict")
def predict_text(text_input: inputdata):
    result = faiss.query_text(text_input.text)
    ret_res = [{"text": d['text'], 'path': d['path']}for d in result] 
    return ret_res


