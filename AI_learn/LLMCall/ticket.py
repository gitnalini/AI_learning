import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is required")

client = Groq(api_key=my_api_key)

class Ticket(BaseModel):
    name: str
    address:str
    email:str
    phone_number:str
    issue:str

schema = Ticket.model_json_schema()
response_format={
    "type": "json_object"
}
system_prompt =f"""Extract the information from the ticket strictly based on this schema: {schema} and give me in json format only. Do not add any extra information or text. If any field is missing in the text, return it as null in the json output. The fields are: name, address, email, phone_number, issue."""

message_system={
    "role":"system",
    "content":system_prompt
}

model = "openai/gpt-oss-120b"
role="user"
text="Hii, My name is John. i broke my nose because of thet and i need to go to bangalore to meet my friends. I have an Iphone which is not working properly. My address is 123, Main Street, New York. Please help me to fix my Iphone, my email is john@example.com and my phone number is 123-456-7890. I have tried restarting it, but it still doesn't work. Can you please assist me in fixing my Iphone? Thank you."
prompt=f""" You are a helpful assistant. Please extract the following information from the text and provide it in a structured format {text}:"""

 
message={
    "role":role,
    "content":prompt
}
messages=[message_system,message]

# Temperature by default is always 0 you can set it to anything you want, but it is recommended to keep it low for better results

response=client.chat.completions.create(model=model, messages=messages)
print("############################################################################################")
answer=response.choices[0].message.content
print(answer) 


# Reading json code 
import json 
raw_json=answer
data_file=json.loads(raw_json) # This will convert the json string into a python dictionary 
ticket=Ticket(**data_file) #This will create a Ticker object from the dictionary

print(ticket.name)
print(ticket.email)
print(ticket.address)
print(ticket.phone_number)
print(ticket.issue)