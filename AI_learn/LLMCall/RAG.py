import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API key is required")

client =Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"


# step 1:knowledge base
knowledge_base={
    "age":"AGE of nal is 22 years",
    "net worth":"Net worth of nal is 1.5 million dollars",
}

# step 2: retrieval function 
def  retrieve_answer(question):
    question=question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net worth"]
    else:
        return "I don't know the answer to that question."

# step 3: RAG function
 
def ask_llm(question):
    context=retrieve_answer(question)
    system_prop=f"""answer in one line only short, answer only based on the context provided donot hallucinate, context"{context}"""
    message_system={
        "role":"system",
        "content":system_prop
    }
    message_user={
        "role":"user",
        "content":question
    }
    messages=[message_system,message_user]
    response=client.chat.completions.create(model=model,messages=messages)
    answer=response.choices[0].message.content
    return answer


question="What is the capital of France?"
print(ask_llm(question))
