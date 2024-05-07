from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from cosine_sim import process_email, adjust_cosine_similarities

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Item(BaseModel):
    text: str

@app.post('/predict')
def predict(item: Item):
    ham_percentage, spam_percentage, indices, cosine_similarities = process_email(item.text)
    return {
        'ham_percentage': ham_percentage, 
        'spam_percentage': spam_percentage,
        'indices': indices,
        'cosine_similarities': cosine_similarities
    }

class Feedback(BaseModel):
    relevant: bool
    indices: List[int]
    cosine_similarities: List[float]


@app.post('/feedback')
def receive_feedback(feedback: Feedback):
    adjusted_cosine_similarities = adjust_cosine_similarities(feedback.indices, feedback.cosine_similarities, feedback.relevant)
    pass