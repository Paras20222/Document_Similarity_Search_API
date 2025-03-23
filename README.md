# Document Similarity Search API

A machine learning powered API that finds similar documents using embeddings and vector search techniques.

## Features

- Convert documents into vector embeddings using Sentence Transformers
- Store embeddings in FAISS (Facebook AI Similarity Search) vector database
- Search for similar documents using various similarity metrics
- Real-time indexing for newly added documents
- RESTful API with FastAPI

## Technical Architecture

1. **Embedding Generation**: Uses Sentence Transformers to convert text documents into vector embeddings
2. **Vector Database**: Uses FAISS for efficient similarity search
3. **API Layer**: FastAPI for exposing search endpoints
4. **Data Storage**: Simple JSON-based storage for document metadata

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/document-similarity-search.git
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

### Option 2: Docker

1. Build the Docker image:
   ```
   docker build -t document-similarity-api .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 document-similarity-api
   ```

## Loading Sample Data

The repository includes a script to load sample BBC news articles:

```
python load_data.py
```

## API Usage

Once the server is running, the API will be available at http://localhost:8000.

### API Endpoints

#### Search for Similar Documents

```
GET /api/search?q=your search query&metric=cosine&limit=5
```

Parameters:
- `q`: Search query (required)
- `metric`: Similarity metric (optional, default: "cosine")
  - Options: "cosine", "dot", "euclidean"
- `limit`: Number of results to return (optional, default: 5)

Example Response:
```json
{
  "results": [
    {
      "document": {
        "id": "42",
        "title": "Technology Article #42",
        "content": "The content of the article...",
        "metadata": {
          "category": "technology",
          "source": "BBC News Dataset"
        }
      },
      "score": 0.8975
    },
    // More results...
  ],
  "query": "your search query",
  "metric": "cosine",
  "total_results": 5
}
```

#### Add a New Document

```
POST /api/documents
```

Request Body:
```json
{
  "title": "Document Title",
  "content": "Document content goes here...",
  "metadata": {
    "category": "example",
    "author": "John Doe"
  }
}
```

#### Get All Documents

```
GET /api/documents?limit=100
```

Parameters:
- `limit`: Maximum number of documents to return (optional, default: 100)

#### Get Document by ID

```
GET /api/documents/{doc_id}
```

## Deployment

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/fastapi)

1. Create a new project on Railway
2. Connect your GitHub repository
3. Railway will automatically detect the Dockerfile and build your app

### Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Extending the Project

### Adding Different Embedding Models

To use a different embedding model, modify the `MODEL_NAME` and `VECTOR_DIMENSION` variables in `app.py`.

### Using a Different Vector Database

The project uses FAISS for vector search, but it can be extended to use other vector databases like:
- Pinecone
- Milvus
- ChromaDB

### Implementing More Advanced Features

- Document chunking for long texts
- Hybrid search (combining vector search with keyword search)
- Query expansion techniques
- User feedback systems

## License

MIT 
