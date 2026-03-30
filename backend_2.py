# backend_rag.py
from fastapi import FastAPI, Request,UploadFile,File
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os
import requests
import json
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import uvicorn

from ultralytics import YOLO
from PIL import Image
from fastapi.responses import JSONResponse
import base64
import cv2
import numpy as np
import io
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

# ----------------------------
# 1️⃣ Load environment variables
# ----------------------------
dotenv_path=r"C:\Users\JBC\Desktop\React Native\ShenAI\.env"
load_dotenv(dotenv_path)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ----------------------------
# 2️⃣ Initialize embeddings
# ----------------------------
# from sentence_transformers import SentenceTransformer

# # Load model
# embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", model=SentenceTransformer("BAAI/bge-small-en-v1.5"))
# Install if needed: pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")  # 1024 dims

#bone fracture

model_path = hf_hub_download(
    repo_id="ZOROD/yolov8-healthcare",
    filename="best.pt",
    repo_type="model"
)


model = YOLO(model_path) # or "runs/detect/train/weights/best.pt"



#mongoDB
uri = "mongodb+srv://Bhavesh:bhavesh@cluster0.u4rgl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    
# ----------------------------
# 3️⃣ Initialize Pinecone and vectorstore
# ----------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "medical-bge"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="original_text"  # this points to your metadata key
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

system_prompt = (
   "You are a medical assistant specialized in factual question answering."
"Use ONLY the information provided in the context below to answer."
"If the answer is not clearly stated in the context, reply: 'The provided information is insufficient.'"

"Guidelines:"
"• Do NOT guess, assume, or hallucinate."
"• Use clear, medically accurate language."
"• Provide a detailed explanation using all relevant context."
"• Write 6–10 sentences."
"• If context contains multiple points, combine them into a long structured answer."

"Context:"
"{context}"

)

query_filter = (
        "You are a strict grammar correction engine. "
        "correct the misspelled words in query and return the corrected query ."
        "Dont provide any extra text ."
        "If the query doesnt contain any errors then provide the same query as output ."
    )



# ----------------------------
# 4️⃣ Ollama query function
# ----------------------------
def query_ollama(messages: List[dict], model="llava:7b-v1.6-mistral-q4_0") -> str:
    """
    Send messages to local Ollama LLava model and return combined response.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
         "num_predict": 1024 
    }

    response = requests.post(url, json=payload, stream=True)
    if response.status_code == 200:
        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        full_text += json_data["message"]["content"]
                except json.JSONDecodeError:
                    continue
        return full_text
    else:
        return f"Request failed: {response.status_code} - {response.text}"

# ----------------------------
# 5️⃣ FastAPI app
# ----------------------------
app = FastAPI(title="RAG + Ollama Backend")

# ----------------------------
# 6️⃣ Request model
# ----------------------------
class QueryRequest(BaseModel):
    query: str

# ----------------------------
# 7️⃣ POST endpoint
# ----------------------------


db=client["MedicalDB"]

collection = db["Shen"]



# @app.post("/data")
# async def process_query(request: QueryRequest):
#     Initial_query = request.query.strip()
    
#     corrected_messages = [
#         {"role": "system", "content": query_filter},
#         {"role": "user", "content": f"User question: {Initial_query}"}
#     ]
    
#     query_text = query_ollama(corrected_messages)

#     # corrected_messages = client.chat.completions.create(
#     #     model="gpt-4o",
#     #     messages=[
#     #         {"role": "system", "content": query_filter},
#     #         {"role": "user", "content": Initial_query}
#     #     ]
#     # )
    
#     # query_text = corrected_messages.choices[0].message.content.strip()
    
#     print(f"corrected query is : {query_text}")

#     # 1️⃣ Retrieve relevant docs from Pinecone
#     docs_and_scores = retriever.vectorstore.similarity_search_with_score(query_text, k=3)

#     # 2️⃣ Filter by score > 0.75
#     filtered_docs = [doc for doc, score in docs_and_scores if score > 0.6]

#     if not filtered_docs:
#     #     answer ="No relevant match found in Pinecone."
#     #     AddDocToDB = [
#     #     {"query": Initial_query, "Answer": answer, "DateTime":datetime.now() }
#     # ]
#     #     result = collection.insert_many(AddDocToDB)
        
#     #     print("Inserted document IDs:", result.inserted_ids)
#         return {"summary": "No relevant match found in Pinecone."}

#     # 3️⃣ Combine context text
    # context_text = "\n".join([doc.page_content for doc in filtered_docs])

    # # 4️⃣ Prepare messages for Ollama
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": f"{context_text}\n\nUser question: {query_text}"}
    # ]

    # # 5️⃣ Query Ollama
    # answer = query_ollama(messages)
    
#     # AddDocToDB = [
#     #     {"query": Initial_query, "Answer": answer, "DateTime":datetime.now() }
#     # ]
#     # result = collection.insert_many(AddDocToDB)
    
#     # print("Inserted document IDs:", result.inserted_ids)
#     # 6️⃣ Return the result
#     return {"summary": answer}


# @app.post("/data")
# async def process_query(request: QueryRequest):
#     query_text = request.query.strip()
#     print(f"Query: {query_text}")

#     docs_and_scores = docsearch.similarity_search_with_score(query_text, k=5)
    
#     print(f"Found {len(docs_and_scores)} documents")
    
#     # Debug both page_content and metadata
#     for i, (doc, score) in enumerate(docs_and_scores):
#         print(f"Doc {i+1} - Score: {score}")
#         print(f"Metadata keys: {list(doc.metadata.keys())}")
#         print(f"Page content: {doc.page_content[:100]}...")
#         print(f"Metadata: {doc.metadata}")
#         print("---")

#     # Filter by score > 0.5
#     filtered_docs = [doc for doc, score in docs_and_scores if score > 0.5]

#     if not filtered_docs:
#         return {"original_texts": []}

#     # Try page_content if metadata is empty
#     original_texts = []
#     for doc in filtered_docs:
#         if doc.metadata and "original_text" in doc.metadata:
#             original_texts.append(doc.metadata["original_text"])
#         else:
#             # Fallback to page_content
#             original_texts.append(doc.page_content)

#     print(f"Retrieved {len(original_texts)} texts after filtering")
#     return {"original_texts": original_texts}


@app.post("/data")
async def process_query(request: QueryRequest):
    query_text = request.query.strip()
    print(f"Query: {query_text}")

    try:
        # Generate embedding for query
        query_embedding = embeddings.embed_query(query_text)
        
        # Query Pinecone directly
        index = pc.Index(index_name)
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        print("Direct Pinecone query results:")
        filtered_texts = []
        
        for i, match in enumerate(results.matches):
            print(f"Result {i+1} - Score: {match.score}")
            
            if match.score > 0.5 and match.metadata:
                # Your text is stored in metadata['text']
                text_content = match.metadata.get("text", "")
                if text_content:
                    filtered_texts.append(text_content)
                    print(f"✓ Included - Text: {text_content[:100]}...")
                else:
                    print("✗ No text found in metadata")
            else:
                print(f"✗ Excluded - Score: {match.score}")
            print("---")
        
        print(f"Final filtered texts: {len(filtered_texts)}")
        
        # context_text = "\n".join([doc.page_content for doc in filtered_texts])

        # # 4️⃣ Prepare messages for Ollama
        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": f"{context_text}\n\nUser question: {query_text}"}
        # ]

        # # 5️⃣ Query Ollama
        # answer = query_ollama(messages)
        
        # return {"summary": answer}
        
        # Return in the format your frontend expects
        if filtered_texts:
            # Join all retrieved texts into one summary
            combined_summary = "\n\n".join(filtered_texts)
            AddDocToDB = [
         {"query": query_text, "Answer": combined_summary, "DateTime":datetime.now() }
             ]
            result = collection.insert_many(AddDocToDB)
    
            print("Inserted document IDs:", result.inserted_ids)
           
            return {"summary": combined_summary}
        else:
            AddDocToDB = [
         {"query": query_text, "Answer": "No relevant information found.", "DateTime":datetime.now() }
             ]
            result = collection.insert_many(AddDocToDB)
    
            print("Inserted document IDs:", result.inserted_ids)
            return {"summary": "No relevant information found."}
        
    except Exception as e:
        print(f"Error: {e}")
        AddDocToDB = [
         {"query": query_text, "Answer": "Error processing request", "DateTime":datetime.now() }
             ]
        result = collection.insert_many(AddDocToDB)
    
        print("Inserted document IDs:", result.inserted_ids)
        return {"summary": "Error processing request"}


# @app.post("/data")
# async def process_query(request: QueryRequest):
#     query_text = request.query.strip()
#     print(f"Query: {query_text}")

#     try:
#         # 1️⃣ Generate embedding
#         query_embedding = embeddings.embed_query(query_text)

#         # 2️⃣ Query Pinecone
#         index = pc.Index(index_name)
#         results = index.query(
#             vector=query_embedding,
#             top_k=25,
#             include_metadata=True
#         )
        
#         print("Direct Pinecone Results:")
#         filtered_texts = []

#         for i, match in enumerate(results.matches):
#             print(f"Result {i+1} - Score: {match.score}")

#             if match.score > 0.3 and match.metadata:
#                 text_content = match.metadata.get("text", "")
#                 if text_content:
#                     filtered_texts.append(text_content)
#                     print(f"✓ Included — {text_content[:80]}...\n")
#                 else:
#                     print("✗ No text field in metadata\n")
#             else:
#                 print("✗ Excluded (Low score)\n")

#         print(f"Total filtered texts: {len(filtered_texts)}")

#         # 3️⃣ Create context correctly
#         context_text = "\n".join(filtered_texts)
#         context_text = context_text[:8000]   # allow more characters


#         # 4️⃣ Prepare messages for Ollama
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {
#                 "role": "user",
#                 "content": f"Context:\n{context_text}\n\nUser question: {query_text}"
#             }
#         ]

#         # 5️⃣ Query Ollama server
#         answer = query_ollama(messages)

#         return {"summary": answer}

#     except Exception as e:
#         print(f"Error: {e}")
#         return {"summary": "Error processing request"}


@app.post("/fractureDetection")
async def fracture_detection(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert to OpenCV (BGR)
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Run YOLO prediction
        results = model.predict(img_cv)

        detections = []
        annotated_frame = img_cv.copy()

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw detection box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{name} {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detections.append({
                "class": name,
                "confidence": round(conf, 3)
            })

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # Send the key name your frontend expects
        return JSONResponse({
            "result": detections if detections else "No fracture detected",
            "output_image": img_base64
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})

