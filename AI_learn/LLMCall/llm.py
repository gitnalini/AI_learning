import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is required")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role="user"
prompt1="HI!!"
prompt2="Explain time travel in 100 or less than 100 tokens"
prompt3="Wrire essay on the importance of AI in modern society 100 or less than 100 tokens"
prompt4="Write essay on Machine learning and its applications 100 or less than 100 tokens"


prompts=[prompt1,prompt2,prompt3,prompt4]

for prompt in prompts:
    message={
        "role":role,
        "content":prompt
    }

    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages, temperature=0.5,max_tokens=200)
    usage=response.usage
    print(f"Prompt: {prompt} -->  completion tokens used: {usage.completion_tokens}, prompt token used:{usage.prompt_tokens}, total token used:{usage.total_tokens}, Finish reason: {response.choices[0].finish_reason}")

# # System : instructions to the model, you can set it to anything you want
# message_system={
#     "role":"system",
#     "content":"You are a brand who suggests clothing company names "
# }

# message={
#     "role":role,
#     "content":prompt
# }
# message=[message_system,message]

# Temperature by default is always 0 you can set it to anything you want, but it is recommended to keep it low for better results

# response=client.chat.completions.create(model=model, messages=message,temperature=2)
# print(response)


# print("############################################################################################")
# answer=response.choices[0].message.content
# print (answer) 