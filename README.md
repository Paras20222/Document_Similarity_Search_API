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
   cd Document_Similarity_Search_API
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
   uvicorn app:app --reload --log-level debug
   ```
5. Access the API
   ```
   Open http://127.0.0.1:8000 in your browser
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

# API Documentation

## Add a Document

### Request
```powershell
PS C:\Users\paras> Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/documents" -Method Post -Headers @{"Content-Type"="application/json"} -Body (@{
    "title"="Google Unveils New AI Model for Search"
    "content"="Google has announced a revolutionary AI-powered search update that enhances user experience by providing more accurate and context-aware results. The model, based on deep learning and natural language understanding, aims to make searching faster and more intuitive."
} | ConvertTo-Json)
```

### Response
```json
{
    "id": "2101",
    "message": "Document added successfully"
}
```

---

## Retrieve Documents

### Request
```powershell
PS C:\Users\paras> Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/search?q=machine%20learning" -Method Get
```

### Response (Server Logs)
```log
INFO:     127.0.0.1:59519 - "GET /api/search?q=machine%20learning HTTP/1.1" 200 OK
2025-03-24 10:38:19,812 - INFO - Search request: q='machine learning', metric=cosine, limit=5, page=1
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 25.11it/s]
INFO: Retrieved document 2094 -> {'id': '2095', 'title': 'AI Revolutionizes Healthcare with Early Disease Detection', 'content': "Recent advancements in artificial intelligence (AI) are transforming the healthcare industry, particularly in early disease detection. Researchers have developed AI algorithms capable of analyzing medical imaging and genetic data to identify diseases such as cancer, Alzheimer's, and heart conditions at much earlier stages. These AI systems, powered by deep learning, can spot patterns that are often invisible to the human eye, leading to faster diagnosis and treatment. With the integration of AI, doctors are now able to offer more personalized treatment plans. The technology is proving to be especially valuable in underserved areas where access to specialists is limited. As AI continues to evolve, it holds the potential to drastically reduce healthcare costs while improving patient outcomes.", 'metadata': {'added_on': '2025-03-23T17:09:05.182Z'}}
INFO: Retrieved document 2093 -> {'id': '2094', 'title': 'Smartphone Technology to Monitor and Influence User Behavior', 'content': "Researchers at Stanford University are working on advanced smartphone systems that track users' routines and behaviors, aiming to predict and subtly influence daily activities, according to TechCrunch.", 'metadata': {'added_on': '2025-03-23T12:23:24.691Z'}}
INFO: Retrieved document 1637 -> {'id': '1638', 'title': 'Behaviour control by smartphone', 'content': 'The Massachusetts Institute of Technology is developing mobile phones designed to learn users #39; daily habits so that they can predict what users will do, reports The Register.', 'metadata': {'category': 4, 'source': 'AG News Dataset'}}
INFO: Retrieved document 467 -> {'id': '468', 'title': 'Deal has S2io champing at the gigabit', 'content': 'OTTAWA -- A local firm that says it can help shrink backup times at large data centres is growing its business thanks to an alliance with Sun Microsystems Inc.', 'metadata': {'category': 3, 'source': 'AG News Dataset'}}
INFO: Retrieved document 904 -> {'id': '905', 'title': '96 Processors Under Your Desktop', 'content': 'Roland Piquepaille writes  quot;A small Santa Clara-based company, Orion Multisystems, today unveils a new concept in computing,  #39;cluster workstations.', 'metadata': {'category': 4, 'source': 'AG News Dataset'}}
```



___________________________________________________________________________________________________________________________________________________________________________________________________________________________



### Implementing More Advanced Features

- Document chunking for long texts
- Hybrid search (combining vector search with keyword search)
- Query expansion techniques
- User feedback systems

