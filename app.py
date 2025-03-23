from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
import uvicorn
import numpy as np
from fastapi.staticfiles import StaticFiles
import os
import json
import logging
import faiss
from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI Initialization
app = FastAPI(
    title="Document Similarity Search API",
    description="Find similar documents using FAISS and SentenceTransformers",
    version="2.0.0"
)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentInput(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    document: Document
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    metric: str
    total_results: int

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
DATA_DIR = "data"
DOCS_FILE = os.path.join(DATA_DIR, "documents.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
METRICS = {"cosine", "dot", "euclidean"}
DEFAULT_METRIC = "cosine"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Global Variables
documents = []
embeddings = np.empty((0, VECTOR_DIM), dtype=np.float32)
model = None
index = None

# Serve the index page
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Index.html not found in static directory")
        return {"error": "Index.html not found"}

# Exception handler for better error messages
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

@app.on_event("startup")
async def load_model_and_data():
    global model, index, documents, embeddings

    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    try:
        if os.path.exists(DOCS_FILE):
            with open(DOCS_FILE, "r") as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} documents")
        else:
            documents = []
            logger.warning(f"Documents file not found at {DOCS_FILE}. Starting with empty collection.")
    except json.JSONDecodeError:
        logger.error("Error loading JSON file: Invalid JSON format")
        documents = []

    try:
        if os.path.exists(EMBEDDINGS_FILE):
            embeddings = np.load(EMBEDDINGS_FILE)
            logger.info(f"Loaded embeddings with shape {embeddings.shape}")
        else:
            embeddings = np.empty((0, VECTOR_DIM), dtype=np.float32)
            logger.warning(f"Embeddings file not found at {EMBEDDINGS_FILE}. Starting with empty embeddings.")
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        embeddings = np.empty((0, VECTOR_DIM), dtype=np.float32)

    await rebuild_index()

async def rebuild_index():
    """Rebuilds FAISS index from stored documents."""
    global index, documents, embeddings

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index = faiss.IndexIDMap(index)

    if not documents:
        logger.info("No documents to index")
        return

    valid_docs = [(i, doc) for i, doc in enumerate(documents) if doc["content"].strip()]
    
    if not valid_docs:
        logger.warning("No valid documents found for indexing")
        return
    
    doc_indices, valid_documents = zip(*valid_docs)
    contents = [doc["content"] for doc in valid_documents]
    doc_ids = np.array([int(doc["id"]) for doc in valid_documents], dtype=np.int64)

    logger.info(f"Encoding {len(contents)} documents...")
    embeddings = np.array(model.encode(contents, show_progress_bar=True), dtype=np.float32)
    
    if embeddings.shape[0] != doc_ids.shape[0]:
        logger.error(f"Mismatch in embeddings ({embeddings.shape[0]}) and IDs ({doc_ids.shape[0]})")
        return

    index.add_with_ids(embeddings, doc_ids)
    logger.info(f"FAISS index rebuilt with {len(doc_ids)} documents")

def compute_similarity(query_embedding, doc_embeddings, metric):
    """Computes similarity scores."""
    if metric == "cosine":
        # Avoid division by zero
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_embedding = np.ones_like(query_embedding) / np.sqrt(len(query_embedding))
        else:
            query_embedding /= query_norm
            
        # Normalize document embeddings
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        doc_embeddings /= norms
        
        return np.dot(doc_embeddings, query_embedding)
    elif metric == "dot":
        return np.dot(doc_embeddings, query_embedding)
    elif metric == "euclidean":
        return np.abs(-np.linalg.norm(doc_embeddings - query_embedding, axis=1))
    raise ValueError(f"Unsupported metric: {metric}")

@app.get("/", tags=["Root"])
async def home():
    return {"message": "Welcome to the Document Similarity Search API", "status": "Running"}

@app.get("/api/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., description="Search query"),
    metric: str = Query(DEFAULT_METRIC, description=f"Similarity metric: {', '.join(METRICS)}"),
    limit: int = Query(5, ge=1, le=20),
    page: int = Query(1, ge=1)
):
    logger.info(f"Search request: q='{q}', metric={metric}, limit={limit}, page={page}")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    if metric not in METRICS:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Must be one of: {', '.join(METRICS)}")

    if not documents or index is None or index.ntotal == 0:
        logger.warning("Search requested but no documents are indexed")
        return SearchResponse(results=[], query=q, metric=metric, total_results=0)

    try:
        query_embedding = model.encode([q])[0]
        scores = compute_similarity(query_embedding, embeddings, metric)
        
        # Get ranked indices by score
        ranked_indices = np.argsort(scores)[::-1]
        
        # Calculate pagination
        start = (page - 1) * limit
        end = start + limit
        
        # Get total results (all matches)
        total_matches = len(ranked_indices)
        
        # Get paginated results
        paginated_indices = ranked_indices[start:end]
        
        results = [
            SearchResult(document=Document(**documents[idx]), score=float(scores[idx]))
            for idx in paginated_indices
        ]
        
        logger.info(f"Search for '{q}' returned {total_matches} total matches")
        return SearchResponse(
            results=results, 
            query=q, 
            metric=metric, 
            total_results=total_matches
        )
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/documents", status_code=201)
async def add_document(document: DocumentInput):
    global documents, embeddings

    # Validate document content
    if not document.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")

    # Generate new document ID
    doc_id = str(max((int(doc["id"]) for doc in documents), default=0) + 1)
    new_doc = {"id": doc_id, **document.dict()}
    documents.append(new_doc)

    # Generate embedding
    try:
        new_embedding = model.encode([document.content])[0].astype(np.float32)
        embeddings = np.vstack([embeddings, new_embedding]) if embeddings.size > 0 else np.array([new_embedding])
    except Exception as e:
        # Roll back document addition if embedding fails
        documents.pop()
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process document")

    # Save updated data
    try:
        np.save(EMBEDDINGS_FILE, embeddings)
        with open(DOCS_FILE, "w") as f:
            json.dump(documents, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        # Continue anyway as the document is already in memory

    # Rebuild index with new document
    await rebuild_index()

    return {"id": doc_id, "message": "Document added successfully"}

@app.get("/api/documents", response_model=List[Document])
async def get_documents(limit: int = Query(100, ge=1, le=1000)):
    return [Document(**doc) for doc in documents[:limit]]

@app.get("/api/documents/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    doc = next((d for d in documents if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return Document(**doc)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "documents_count": len(documents),
        "embeddings_shape": embeddings.shape if hasattr(embeddings, "shape") else None,
        "index_size": index.ntotal if index is not None else 0
    }

# Add a debugging endpoint to check frontend-backend communication
@app.get("/api/debug/echo")
async def echo_request(q: str = Query(None)):
    return {
        "received_query": q,
        "documents_count": len(documents),
        "embeddings_shape": embeddings.shape if hasattr(embeddings, "shape") else None
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
