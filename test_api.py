

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000/api"
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/documents"

def print_separator():
    print("\n" + "="*80 + "\n")

def test_add_document():
    print("Testing: Add a new document")
    
    # Sample document
    document = {
        "title": "Artificial Intelligence in Healthcare",
        "content": "Artificial intelligence (AI) is revolutionizing healthcare in numerous ways. " +
                   "From diagnosing diseases to personalized treatment plans, AI systems are " +
                   "helping physicians make better decisions faster. Machine learning algorithms " +
                   "can analyze medical images, predict patient outcomes, and identify potential " +
                   "health risks before they become serious problems.",
        "metadata": {
            "category": "technology",
            "tags": ["AI", "healthcare", "machine learning"]
        }
    }
    
    # Send the request
    response = requests.post(DOCUMENTS_ENDPOINT, json=document)
    
    # Print results
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json().get("id")

def test_search(query, metric="cosine"):
    print(f"Testing: Search for '{query}' using {metric} similarity")
    
    # Send the request
    params = {
        "q": query,
        "metric": metric,
        "limit": 5
    }
    
    response = requests.get(SEARCH_ENDPOINT, params=params)
    
    # Print results
    print(f"Status Code: {response.status_code}")
    print("Top Results:")
    
    if response.status_code == 200:
        data = response.json()
        for i, result in enumerate(data["results"]):
            print(f"\n{i+1}. {result['document']['title']} (Score: {result['score']:.4f})")
            print(f"   ID: {result['document']['id']}")
            content_preview = result['document']['content'][:100] + "..."
            print(f"   Content: {content_preview}")
    else:
        print(f"Error: {response.text}")

def test_get_documents():
    print("Testing: Get all documents")
    
    response = requests.get(DOCUMENTS_ENDPOINT, params={"limit": 10})
    
    # Print results
    print(f"Status Code: {response.status_code}")
    print(f"Found {len(response.json())} documents")
    
    if response.status_code == 200 and len(response.json()) > 0:
        doc = response.json()[0]
        print("\nSample document:")
        print(f"ID: {doc['id']}")
        print(f"Title: {doc['title']}")
        content_preview = doc['content'][:100] + "..."
        print(f"Content: {content_preview}")

def test_get_document_by_id(doc_id):
    print(f"Testing: Get document by ID ({doc_id})")
    
    response = requests.get(f"{DOCUMENTS_ENDPOINT}/{doc_id}")
    
    # Print results
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        doc = response.json()
        print(f"Title: {doc['title']}")
        content_preview = doc['content'][:100] + "..."
        print(f"Content: {content_preview}")
    else:
        print(f"Error: {response.text}")

def test_different_metrics():
    query = "artificial intelligence technology"
    
    for metric in ["cosine", "dot", "euclidean"]:
        test_search(query, metric)
        print_separator()

def main():
    print("Document Similarity Search API - Test Script")
    print_separator()
    
    # Test adding a document
    doc_id = test_add_document()
    print_separator()
    
    # Wait for the embedding to be processed
    print("Waiting for document processing...")
    time.sleep(2)
    
    # Test get document by ID
    if doc_id:
        test_get_document_by_id(doc_id)
        print_separator()
    
    # Test get all documents
    test_get_documents()
    print_separator()
    
    # Test search with different queries
    test_search("artificial intelligence")
    print_separator()
    
    test_search("healthcare and medicine")
    print_separator()
    
    # Test different similarity metrics
    test_different_metrics()

if __name__ == "__main__":
    main() 
