from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict, Any, Optional
import uvicorn
import numpy as np
from fastapi.staticfiles import StaticFiles
import os
import json
import logging
import faiss
from sentence_transformers import SentenceTransformer
import time

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
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {str(e)}")

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
    time_ms: float

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
DATA_DIR = "data"
DOCS_FILE = os.path.join(DATA_DIR, "documents.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings_.npy")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index_.index")
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
        return HTMLResponse(content="<html><body><h1>Welcome to Document Similarity Search API</h1><p>HTML interface not found. Please use the API endpoints directly.</p></body></html>")

# Exception handler for better error messages
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

def save_faiss_index(index_to_save, filepath):
    """Save FAISS index to file"""
    try:
        faiss.write_index(index_to_save, filepath)
        logger.info(f"FAISS index saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {str(e)}")
        return False

def load_faiss_index(filepath):
    """Load FAISS index from file"""
    try:
        if os.path.exists(filepath):
            loaded_index = faiss.read_index(filepath)
            logger.info(f"FAISS index loaded from {filepath}")
            return loaded_index
        else:
            logger.warning(f"No FAISS index found at {filepath}")
            return None
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {str(e)}")
        return None

def compute_similarity(query_embedding, doc_embeddings, metric="cosine", normalize=True):
    if normalize:
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    if metric == "cosine":
        scores = np.dot(doc_embeddings, query_embedding)
        scores = (scores + 1) / 2
    
    elif metric == "dot":
        scores = np.dot(doc_embeddings, query_embedding)
        scores = (scores + 1) / 2
    
    elif metric == "euclidean":
        distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
        scores = 1 / (1 + distances)
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return scores

@app.on_event("startup")
async def load_model_and_data():
    global model, index, documents, embeddings

    logger.info(f"Loading model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model = None
        # We'll continue anyway and handle the error in the endpoints

    # Load documents
    try:
        if os.path.exists(DOCS_FILE):
            with open(DOCS_FILE, "r") as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} documents from {DOCS_FILE}")
        else:
            documents = []
            logger.warning(f"Documents file not found at {DOCS_FILE}. Starting with empty collection.")
    except json.JSONDecodeError:
        logger.error("Error loading JSON file: Invalid JSON format")
        documents = []
    except Exception as e:
        logger.error(f"Unexpected error loading documents: {str(e)}")
        documents = []

    # Load embeddings
    try:
        if os.path.exists(EMBEDDINGS_FILE):
            embeddings = np.load(EMBEDDINGS_FILE)
            logger.info(f"Loaded embeddings with shape {embeddings.shape} from {EMBEDDINGS_FILE}")
        else:
            embeddings = np.empty((0, VECTOR_DIM), dtype=np.float32)
            logger.warning(f"Embeddings file not found at {EMBEDDINGS_FILE}. Starting with empty embeddings.")
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        embeddings = np.empty((0, VECTOR_DIM), dtype=np.float32)

    # Try to load existing FAISS index
    index = load_faiss_index(INDEX_FILE)
    
    # If index couldn't be loaded or document counts don't match, rebuild it
    if model is not None and (index is None or (index.ntotal != len(documents))):
        logger.info("Index needs to be rebuilt")
        await rebuild_index()
    else:
        logger.info(f"FAISS index ready with {index.ntotal if index else 0} vectors")

async def rebuild_index():
    """Rebuilds FAISS index from stored documents."""
    global index, documents, embeddings

    if model is None:
        logger.error("Cannot rebuild index: Model not loaded")
        return
        
    try:
        # Initialize a new index
        logger.info("Creating new FAISS index")
        index = faiss.IndexFlatL2(VECTOR_DIM)
        index = faiss.IndexIDMap(index)

        if not documents:
            logger.info("No documents to index")
            return

        valid_docs = [(i, doc) for i, doc in enumerate(documents) if doc.get("content", "").strip()]
        
        if not valid_docs:
            logger.warning("No valid documents found for indexing")
            return
        
        doc_indices, valid_documents = zip(*valid_docs)
        contents = [doc["content"] for doc in valid_documents]
        
        # Create document IDs
        try:
            doc_ids = np.array([int(doc["id"]) for doc in valid_documents], dtype=np.int64)
        except (ValueError, KeyError) as e:
            logger.error(f"Error processing document IDs: {str(e)}")
            # Generate sequential IDs instead
            doc_ids = np.array(range(len(valid_documents)), dtype=np.int64)
            # Update documents with new IDs
            for i, doc_idx in enumerate(doc_indices):
                documents[doc_idx]["id"] = str(i)

        logger.info(f"Encoding {len(contents)} documents...")
        start_time = time.time()
        embeddings = np.array(model.encode(contents, show_progress_bar=True), dtype=np.float32)
        logger.info(f"Encoding completed in {time.time() - start_time:.2f} seconds")
        
        if embeddings.shape[0] != doc_ids.shape[0]:
            logger.error(f"Mismatch in embeddings ({embeddings.shape[0]}) and IDs ({doc_ids.shape[0]})")
            return

        # Add vectors to the index
        index.add_with_ids(embeddings, doc_ids)
        logger.info(f"FAISS index rebuilt with {len(doc_ids)} documents")
        
        # Save updated data
        try:
            np.save(EMBEDDINGS_FILE, embeddings)
            with open(DOCS_FILE, "w") as f:
                json.dump(documents, f, indent=2)
            logger.info(f"Saved {len(documents)} documents and {embeddings.shape[0]} embeddings")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
        
        # Save the index to file
        save_faiss_index(index, INDEX_FILE)
    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")

@app.get("/api", tags=["Root"])
async def home():
    return {
        "message": "Welcome to the Document Similarity Search API", 
        "status": "Running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/api/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., description="Search query"),
    metric: str = Query(DEFAULT_METRIC, description=f"Similarity metric: {', '.join(METRICS)}"),
    limit: int = Query(5, ge=1, le=20),
    page: int = Query(1, ge=1),
    normalize: bool = Query(True, description="Normalize similarity scores to [0,1] range")
):
    start_time = time.time()
    logger.info(f"Search request: q='{q}', metric={metric}, limit={limit}, page={page}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    if metric not in METRICS:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Must be one of: {', '.join(METRICS)}")

    if not documents or index is None or index.ntotal == 0:
        logger.warning("Search requested but no documents are indexed")
        return SearchResponse(
            results=[], 
            query=q, 
            metric=metric, 
            total_results=0,
            time_ms=0
        )

    try:
        # Encode query
        query_embedding = model.encode([q])[0]
        
        # Compute similarity scores
        scores = compute_similarity(query_embedding, embeddings, metric, normalize=normalize)
        
        # Get ranked indices by score
        ranked_indices = np.argsort(scores)[::-1]
        
        # Calculate pagination
        start = (page - 1) * limit
        end = start + limit
        
        # Get total results (all matches)
        total_matches = len(ranked_indices)
        
        # Get paginated results
        paginated_indices = ranked_indices[start:end]
        
        results = []
        
        for idx in paginated_indices:
            if idx < len(documents) and idx < len(scores):
                doc = documents[idx]
                # Ensure document has required fields
                if not all(k in doc for k in ["id", "title", "content"]):
                    logger.warning(f"Document at index {idx} missing required fields")
                    continue
                    
                results.append(
                    SearchResult(
                        document=Document(**doc),
                        score=float(scores[idx])
                    )
                )
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.info(f"Search for '{q}' returned {total_matches} total matches in {elapsed_time:.2f}ms")
        
        return SearchResponse(
            results=results, 
            query=q, 
            metric=metric, 
            total_results=total_matches,
            time_ms=elapsed_time
        )
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/documents", status_code=201)
async def add_document(document: DocumentInput):
    global documents, embeddings, index  # Ensure index is in scope
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

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
        
        # Add to embeddings array
        if embeddings.size > 0:
            embeddings = np.vstack([embeddings, new_embedding])
        else:
            embeddings = np.array([new_embedding])
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

    # Add to index directly if it exists
    if index is not None:
        try:
            # Reshape to ensure 2D array for FAISS
            new_embedding_reshaped = new_embedding.reshape(1, -1)
            index.add_with_ids(new_embedding_reshaped, np.array([int(doc_id)], dtype=np.int64))
            save_faiss_index(index, INDEX_FILE)
        except Exception as e:
            logger.error(f"Failed to update index: {str(e)}")
            # Rebuild the whole index as fallback
            await rebuild_index()
    else:
        # Rebuild index if it doesn't exist
        await rebuild_index()

    return {"id": doc_id, "message": "Document added successfully"}

@app.delete("/api/documents/{doc_id}", status_code=200)
async def delete_document(doc_id: str):
    global documents, embeddings, index
    
    # Find document index
    doc_idx = next((i for i, doc in enumerate(documents) if doc["id"] == doc_id), None)
    
    if doc_idx is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from documents list
    documents.pop(doc_idx)
    
    # Remove from embeddings array
    if doc_idx < len(embeddings):
        embeddings = np.delete(embeddings, doc_idx, axis=0)
    
    # Save updated data
    try:
        np.save(EMBEDDINGS_FILE, embeddings)
        with open(DOCS_FILE, "w") as f:
            json.dump(documents, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save data after deletion: {str(e)}")
    
    # Rebuild index (FAISS doesn't support efficient removal)
    await rebuild_index()
    
    return {"message": f"Document {doc_id} deleted successfully"}

@app.get("/api/documents", response_model=List[Document])
async def get_documents(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    if not documents:
        return []
        
    end = min(offset + limit, len(documents))
    if offset >= len(documents):
        return []
        
    return [Document(**doc) for doc in documents[offset:end]]

@app.get("/api/documents/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    doc = next((d for d in documents if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return Document(**doc)

@app.get("/health")
async def health_check():
    model_status = "loaded" if model is not None else "not_loaded"
    index_status = "loaded" if index is not None else "not_loaded"
    
    return {
        "status": "healthy" if model is not None else "degraded",
        "documents_count": len(documents),
        "embeddings_shape": embeddings.shape if hasattr(embeddings, "shape") else None,
        "index_size": index.ntotal if index is not None else 0,
        "model_status": model_status,
        "index_status": index_status,
        "metrics_available": list(METRICS)
    }

@app.get("/api/metrics")
async def get_metrics():
    """Return available similarity metrics"""
    return {
        "available_metrics": list(METRICS),
        "default_metric": DEFAULT_METRIC
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
