# backend_rag.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from datetime import datetime

import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse

# ----------------------------
# Load ENV
# ----------------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
MONGO_URI = "mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Optimized RAG Backend 🚀")

# ----------------------------
# Request Model
# ----------------------------
class QueryRequest(BaseModel):
    query: str


# ============================
# 🔁 LAZY LOAD FUNCTIONS
# ============================

def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"  # smaller + faster
    )


def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index("medical-bge")


def get_mongo_collection():
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    return client["MedicalDB"]["Shen"]


def get_yolo_model():
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="ZOROD/yolov8-healthcare",
        filename="best.pt"
    )
    return YOLO(model_path)


# ============================
# 📊 RAG ENDPOINT
# ============================

@app.post("/data")
async def process_query(request: QueryRequest):
    query_text = request.query.strip()
    print(f"Query: {query_text}")

    try:
        embeddings = get_embeddings()
        index = get_pinecone_index()
        collection = get_mongo_collection()

        # Generate embedding
        query_embedding = embeddings.embed_query(query_text)

        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )

        filtered_texts = []

        for match in results.matches:
            if match.score > 0.5 and match.metadata:
                text = match.metadata.get("text", "")
                if text:
                    filtered_texts.append(text)

        if filtered_texts:
            combined = "\n\n".join(filtered_texts)
        else:
            combined = "No relevant information found."

        # Save to MongoDB
        collection.insert_one({
            "query": query_text,
            "answer": combined,
            "time": datetime.now()
        })

        return {"summary": combined}

    except Exception as e:
        print("ERROR:", e)
        return {"summary": "Error processing request"}


# ============================
# 🦴 FRACTURE DETECTION
# ============================

@app.post("/fractureDetection")
async def fracture_detection(file: UploadFile = File(...)):
    try:
        model = get_yolo_model()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model.predict(img)

        detections = []
        annotated = img.copy()

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{name} {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detections.append({
                "class": name,
                "confidence": round(conf, 3)
            })

        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse({
            "result": detections if detections else "No fracture detected",
            "output_image": img_base64
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})


# ============================
# ROOT (IMPORTANT FOR RENDER)
# ============================

@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}
