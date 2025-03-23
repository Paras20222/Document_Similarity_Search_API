document.addEventListener('DOMContentLoaded', function () {
    // Elements
    const queryInput = document.getElementById('query');
    const metricSelect = document.getElementById('metric');
    const resultsDiv = document.getElementById('results');
    const searchButton = document.getElementById('searchButton');
    const addDocButton = document.getElementById('addDocButton');
    const docTitleInput = document.getElementById('docTitle');
    const docContentInput = document.getElementById('docContent');
     


    // Search documents function
    function searchDocuments() {
        const query = queryInput.value.trim();
        const metric = metricSelect.value;

        if (!query) {
            showMessage('Please enter a search query', 'error');
            return;
        }

        // Clear previous results and show loading
        resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

        fetch(`http://127.0.0.1:8000/api/search?q=${encodeURIComponent(query)}&metric=${metric}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                showMessage(`Error: ${error.message}`, 'error');
            });
    }

    // Display search results
    function displayResults(data) {
        resultsDiv.innerHTML = '';

        // Create results header
        const header = document.createElement('div');
        header.classList.add('results-header');
        header.innerHTML = `
            <h2>Search Results</h2>
            <p>Query: "${data.query}" using ${data.metric} similarity</p>
        `;
        resultsDiv.appendChild(header);

        // Check if we have results
        if (data.results && data.results.length > 0) {
            data.results.forEach((result, index) => {
                const doc = result.document;
                const score = result.score;

                const resultItem = document.createElement('div');
                resultItem.classList.add('result-item');

                let content = doc.content;
                if (content.length > 400) {
                    content = content.substring(0, 400) + '...';
                }

                resultItem.innerHTML = `
                    <div class="result-header">
                        <h3>${index + 1}. ${doc.title || 'Untitled Document'}</h3>
                        <span class="score">Score: ${score.toFixed(4)}</span>
                    </div>
                    <p class="doc-content">${content}</p>
                    <p class="doc-id">Document ID: ${doc.id}</p>
                `;

                resultsDiv.appendChild(resultItem);
            });
        } else {
            showMessage('No similar documents found.', 'info');
        }
    }

    // Add new document function
    function addDocument() {
        const title = docTitleInput.value.trim();
        const content = docContentInput.value.trim();

        if (!title) {
            showMessage('Please enter a document title', 'error');
            return;
        }

        if (!content) {
            showMessage('Please enter document content', 'error');
            return;
        }

        // Show loading
        showMessage('Adding document...', 'info');

        fetch('http://127.0.0.1:8000/api/documents', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: title,
                content: content,
                metadata: {
                    added_on: new Date().toISOString()
                }
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            showMessage(`Document added successfully with ID: ${data.id}`, 'success');
            docTitleInput.value = '';
            docContentInput.value = '';
        })
        .catch(error => {
            showMessage(`Error adding document: ${error.message}`, 'error');
        });
    }

    // Helper function to show messages
    function showMessage(message, type = 'info') {
        resultsDiv.innerHTML = `<div class="message ${type}">${message}</div>`;
    }

    // Event listeners
    searchButton.addEventListener('click', searchDocuments);
    addDocButton.addEventListener('click', addDocument);

    // Handle Enter key in search box
    queryInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            searchDocuments();
        }
    });

    // Initial check for documents
    fetch('http://127.0.0.1:8000/health')
        .then(response => response.json())
        .then(data => {
            if (data.documents_count === 0) {
                showMessage('No documents in the database. Add some documents to get started!', 'info');
            }
        })
        .catch(error => {
            console.error('Error checking API health:', error);
        });
});
