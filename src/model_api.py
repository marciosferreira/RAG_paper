from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

try:
    from models import EmbeddingModel, LLMGenerator
except:
    from .models import EmbeddingModel, LLMGenerator

app = FastAPI(title="ML Model API", version="1.0")
embedd_model = EmbeddingModel()
llm_instruct = LLMGenerator()

class TextInput(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    content: str

class LLMInputs(BaseModel):
    text: List[Message]

    
@app.get("/")
def read_root():
    return {"message": "Model API is up and running!"}

@app.post("/embedding")
def predict_embedding(input: TextInput):
    try:
        prediction = embedd_model(input.text)
        assert isinstance(prediction, list)
        return {"embedding": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def llm(input: LLMInputs):
    try:
        inp = []
        for t in input.text:
            inp.append({'role': t.role, 'content': t.content})
        prediction = llm_instruct(inp)
        return {'content': prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
