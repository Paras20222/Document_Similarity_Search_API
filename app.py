import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Constants
VECTOR_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"
DOCUMENTS_CHUNKS_DIR = "documents_chunks"  # Directory where chunked files are stored

# Global Variables
documents = []
embeddings = np.empty((0, VECTOR_DIM), dtype=np.float32)
model = None
index = None

async def rebuild_index():
    """Rebuilds FAISS index from chunked document files."""
    global index, documents, embeddings

    # Initialize model if not already loaded
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    # Create or clear the FAISS index
    if faiss.get_num_gpus() > 0:
        # Use GPU if available
        index = faiss.IndexFlatL2(VECTOR_DIM)
        index = faiss.IndexIDMap(index)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatL2(VECTOR_DIM)
        index = faiss.IndexIDMap(index)

    # Load documents from chunked files in the "documents_chunks" directory
    chunk_files = [f for f in os.listdir(DOCUMENTS_CHUNKS_DIR) if f.endswith(".json")]
    chunk_files.sort()  # Optional: to process the chunks in a sorted order

    # Load each chunk and add to the FAISS index
    for chunk_file in chunk_files:
        chunk_path = os.path.join(DOCUMENTS_CHUNKS_DIR, chunk_file)
        logger.info(f"Loading chunk: {chunk_path}")

        try:
            with open(chunk_path, "r") as f:
                chunk_documents = json.load(f)

                # Process in smaller batches to avoid memory issues
                batch_size = 2500
                for i in range(0, len(chunk_documents), batch_size):
                    batch = chunk_documents[i:i + batch_size]

                    # Extract content and prepare embeddings
                    contents = [doc["content"] for doc in batch]
                    embeddings_batch = np.array(model.encode(contents, show_progress_bar=True), dtype=np.float32)

                    # Prepare document IDs (based on the index of documents in the batch)
                    doc_ids = np.array([int(doc["id"]) for doc in batch], dtype=np.int64)

                    # Add embeddings to the FAISS index
                    if embeddings_batch.shape[0] != doc_ids.shape[0]:
                        logger.error(f"Mismatch in number of embeddings ({embeddings_batch.shape[0]}) and document IDs ({doc_ids.shape[0]})")
                        continue

                    index.add_with_ids(embeddings_batch, doc_ids)
                    logger.info(f"Added {len(batch)} documents from {chunk_file} to index")

        except Exception as e:
            logger.error(f"Error loading chunk {chunk_file}: {str(e)}")

    logger.info(f"FAISS index rebuilt with {index.ntotal} documents")

    # Save the rebuilt index to file
    try:
        faiss.write_index(index, "faiss_index.index")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {str(e)}")




def compute_similarity(query_embedding, doc_embeddings, metric):
    """Computes similarity scores."""
    if metric == "cosine":
        # Add your cosine similarity logic here
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
        # Normalize vectors for comparable scores
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_embedding = np.ones_like(query_embedding) / np.sqrt(len(query_embedding))
        else:
            query_embedding /= query_norm
            
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        doc_embeddings /= norms
        
        return np.dot(doc_embeddings, query_embedding)
    
    elif metric == "euclidean":
        # Calculate Euclidean distance and convert to similarity score
        distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
        # Invert distance to get similarity score (higher is more similar)
        max_distance = np.max(distances)
        if max_distance == 0:
            return np.ones_like(distances)
        else:
            return 1 - (distances / max_distance)
    
    raise ValueError(f"Unsupported metric: {metric}")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Index.html not found in static directory")
        return {"error": "Index.html not found"}

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

    # Update index with new document
    if faiss.get_num_gpus() > 0:
        # Use GPU if available
        new_index = faiss.IndexFlatL2(VECTOR_DIM)
        new_index = faiss.IndexIDMap(new_index)
        res = faiss.StandardGpuResources()
        new_index = faiss.index_cpu_to_gpu(res, 0, new_index)
    else:
        new_index = faiss.IndexFlatL2(VECTOR_DIM)
        new_index = faiss.IndexIDMap(new_index)

    new_index.add_with_ids([new_embedding], [int(doc_id)])

    # Merge new index with existing index
    if index is not None:
        if faiss.get_num_gpus() > 0:
            index.merge_from(new_index)
        else:
            index.add_with_ids([new_embedding], [int(doc_id)])
    else:
        index = new_index

    logger.info(f"Added document {doc_id} to index")

    return {"id": doc_id, "message": "Document added successfully"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
