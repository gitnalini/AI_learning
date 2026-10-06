import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

my_api_key = os.getenv("GROQ_API_KEY")
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")

if not my_api_key:
    raise ValueError("NO GROQ API KEY AVAILABLE")

if not QDRANT_URL:
    raise ValueError("NO QDRANT URL AVAILABLE")

if not QDRANT_API_KEY:
    raise ValueError("NO QDRANT API KEY AVAILABLE")

 
# connecting to qdrant client

client=QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
print("connected to qdrant client")

# create qdrant collection 

COLLECTION_NAME="knowledge"
EMBEDDING_SIZE=384

# delete collection if exists 

if client.collection_exists(COLLECTION_NAME):
    print(f"Collection {COLLECTION_NAME} already exists. Deleting it...")
    client.delete_collection(COLLECTION_NAME)

# create collection 

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"Collection {COLLECTION_NAME} created")
print(f"Collection {COLLECTION_NAME}, Vector size: {EMBEDDING_SIZE}")
print(f"Collection {COLLECTION_NAME},  distance: {Distance.COSINE}")


# load knowledge 

with open(BASE_DIR / "knowledge.txt", "r") as f:
    documents=[
        line.strip()
        for line in f
        if line.strip()
    ]
print(f"Loaded {len(documents)} documents")

# create embedding model
print("Creating embedding model...")
model =SentenceTransformer('all-MiniLM-L6-v2') #based on 384 features

print("Embedding model created")

embeddings=model.encode(documents)
print(f"Embedding created for {len(documents)} documents")
print(f"Embedding size: {len(embeddings[0])}")

# create qdrant points(id vector payload)

points=[]

for i, embedding in enumerate(embeddings):

    point=PointStruct(
        id=i+1,
        vector=embedding.tolist(),
        payload={
            "text":documents[i]
        }
    )
    points.append(point)

# upload to qdrant

client.upsert( #upload + insert
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"uploaded {len(points)} doocuments to qdrant")

# search in qdrant

def search(query, top_k=3):
    query_vector=model.encode(query).tolist()

    # search qdrant for similar docs
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return results


# test search

query="How many days off do I get"
results=search(query,top_k=1)
for result in results:
    print(f"score: {result.score:.3f}")
    print(f"text: {result.payload['text']}")

# conect to groq client 

groq_client=Groq(api_key=my_api_key)
client_model="openai/gpt-oss-120b"

def ask_llm(user_prompt,context):
    system_prompt=f"""answer in one line only. Answer based on the context only, donot hallucinate, context:{context}"""
    message_system={
        "role":"system",
        "content":system_prompt,
    }
    user_message={
        "role":"user",
        "content":user_prompt,
    }
    messages=[message_system,user_message]
    stream=groq_client.chat.completions.create(
        model=client_model,
        messages=messages,
        stream=True
    )
    answer=""
    for chunk in stream:
        content=chunk.choices[0].delta.content or ""
        print(content, end="")
        answer+=content
    return answer

context = "\n".join(
    result.payload["text"]
    for result in results
)
answer=ask_llm(query,context)











