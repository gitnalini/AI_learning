import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np



model=SentenceTransformer('all-MiniLM-L6-v2') #based on 384 features
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("NOT GROQ API KEY AVAIALABLE")

client=Groq(api_key=my_api_key)
groq_model = "openai/gpt-oss-120b"

def cosine_similarity(vec1, vec2):
    return vec1 @ vec2.T / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

documents = [
    "Employees receive 24 days of paid leave per year.",
   
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
   
    "Employees receive Rs 3000 per month for gym reimbursement.",
   
    "Employees can claim Rs 2000 per month for home internet.",
   
    "Employees have a 90 day notice period."
]

embed=model.encode(documents)


def retrieve_answer(qembedding):
    scores=[]
    for i,document in enumerate(embed):
        score=cosine_similarity(qembedding,document) #returns a score and context that is to be shared to the ask_llm function
        scores.append((score,documents[i]))
    scores.sort(reverse=True)
    return scores[0][0],scores[0][1] #returning the context and score of the most relevant document

def ask_llm(user_prompt,context):
    
    system_promt=f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
        
    system={
        "role":"system",
        "content":system_promt
    }
    user={
        "role":"user",
        "content":user_prompt
    }
    messages=[system,user]
    stream = client.chat.completions.create(
        model=groq_model,
        messages=messages,
        stream=True
    )

    answer = ""

    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        print(content, end="")
        answer += content

    return answer

user_query="How many vacation days do employees get?"

qembedding=model.encode(user_query)
score,context=retrieve_answer(qembedding)
answer=ask_llm(user_query,context)
 