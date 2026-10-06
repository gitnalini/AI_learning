import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(vec1, vec2):
    return vec1 @ vec2.T / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

model=SentenceTransformer('all-MiniLM-L6-v2') #based on 384 features
text="MACHINE LEARNING IS FUN AND EXCITING"

# embedding=model.encode(text)
# print("Embedding shape:", embedding.shape)
# print("Embedding:", embedding[:20]) # Print the first 20 elements of the embedding


t1="There are 25 people in the room."
t2="There are 500 people in the room."

v1=model.encode(t1)
v2=model.encode(t2)

print("Cosine similarity between t1 and t2:", cosine_similarity(v1, v2))