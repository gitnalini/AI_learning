from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is required")

client=Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

def llm(prompt):
    message={
        "role":"user",
        "content":prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model,messages=messages)
    ans=response.choices[0].message.content
    return ans

bad_prompt="""
# ROLE
You are a electronics comapny assistance 
#TASK
Your task is to classify the issue in category
# CONSTRAINT
You have to classify the issue into 3 category Billing,Technical,Return
# OUTPUT FORMAT
The output should be in one world only and should be from the three category
# ONE SHOT 
For example: If somne says broke my leg classify it as Others
# FALLBACK
If the issue is unrelated to any of the categories mentioned in constraints, then the answer should be OTHER
This is a user complaint:
I got wrong food deliver
"""
print(llm(bad_prompt))
