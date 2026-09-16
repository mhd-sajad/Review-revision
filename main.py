from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np

class no(BaseModel):
    num : int

model = joblib.load('model.pkl')

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Model is working perfectly"}

@app.post('/predict')
def pred(data:no):
    inp = np.array([[data.num]])
    pred = (model.predict(inp)).item()
    return {"result":pred}