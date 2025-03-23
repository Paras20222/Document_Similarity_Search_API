# Document Similarity Search API

A machine learning powered API that finds similar documents using embeddings and vector search techniques.

## Features

- Convert documents into vector embeddings using Sentence Transformers
- Store embeddings in FAISS (Facebook AI Similarity Search) vector database
- Search for similar documents using various similarity metrics
- Real-time indexing for newly added documents

## Technical Architecture

1. **Embedding Generation**: Uses Sentence Transformers to convert text documents into vector embeddings
2. **Vector Database**: Uses FAISS for efficient similarity search
3. **API Layer**: FastAPI for exposing search endpoints
4. **Data Storage**: Simple JSON-based storage for document metadata

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/Paras20222/Document_Similarity_Search_API
   cd document-similarity-search
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   uvicorn app:app --reload
   ```

## Loading Sample Data

The repository includes a script to load sample AG news dataset:

```
python load_data.py
```
## Sample Videos

You can also view these two sample videos related to the project:

- [Similarity Search](Video_1.mp4)
- [Real Time Indexing](Video_2.mp4)

### Implementing More Advanced Features

- Document chunking for long texts
- Hybrid search (combining vector search with keyword search)
- Query expansion techniques
- User feedback systems

