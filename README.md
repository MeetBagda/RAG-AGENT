# RAG-Powered Knowledge Assistant

A Retrieval-Augmented Generation (RAG) based Q&A system that provides accurate answers by combining document retrieval with state-of-the-art language models. This system is designed to efficiently process and answer questions about company documentation, FAQs, and technical specifications.

## Features

- 📚 Document Processing
  - Automatic document chunking and indexing
  - Smart content categorization
  - Support for multiple document formats (.txt files)

- 🔍 Vector-based Search
  - FAISS similarity search
  - Efficient document retrieval
  - Context-aware responses

- 🤖 AI-Powered Question Answering
  - Uses Hugging Face's T5 model for natural language understanding
  - Context-aware response generation
  - Handles complex queries with proper context

- 🌐 Web Interface
  - Clean and intuitive Streamlit interface
  - Real-time response generation
  - Debug information in sidebar

## Prerequisites

- Python 3.8 or higher
- Windows/Linux/MacOS
- 4GB RAM minimum (8GB recommended)
- Internet connection (for initial model download)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG AGENT
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
RAG AGENT/
├── data/                      # Document storage
│   └── company_docs/          # Company documentation
│       ├── api_documentation.txt
│       ├── company_faq.txt
│       └── product_specs.txt
├── src/                       # Source code
│   ├── agent.py              # QA agent implementation
│   ├── app.py                # Streamlit web interface
│   ├── document_processor.py # Document processing
│   └── vector_store.py       # Vector store operations
├── tests/                    # Test files
│   ├── test_agent.py
│   ├── test_document_processor.py
│   └── test_vector_store.py
├── requirements.txt          # Project dependencies
└── run_tests.py             # Test runner script
```

## Running the Application

1. Make sure you're in the virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/MacOS
   source venv/bin/activate
   ```

2. Start the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Testing

Run the test suite to verify everything is working correctly:
```bash
python run_tests.py
```

All tests should pass with output similar to:
```
test_load_documents ... ok
test_process_documents ... ok
test_create_vector_store ... ok
test_similarity_search ... ok
...
```

## Sample Questions

Try these example questions to test the system:

1. "What industries does the company serve?"
2. "What are the key features of AI Assistant Pro?"
3. "What is the company's mission?"
4. "What are the technical requirements for the product?"

## Debug Information

- Check the sidebar in the Streamlit interface for:
  - Number of documents loaded
  - Document processing statistics
  - Force retrain option

## Troubleshooting

1. If you get model loading errors:
   - Ensure you have a stable internet connection
   - Try forcing a retrain using the sidebar button

2. If documents aren't being found:
   - Check that your documents are in the correct format
   - Verify files are in the `data/company_docs/` directory
   - Use the Force Retrain button in the sidebar

3. Memory issues:
   - Ensure you have at least 4GB of free RAM
   - Close other memory-intensive applications

## Implementation Details

- Document Processing: Uses RecursiveCharacterTextSplitter with optimized chunk sizes
- Embeddings: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- Vector Store: FAISS for efficient similarity search
- Language Model: google/flan-t5-small for optimal performance
- Frontend: Streamlit with real-time processing

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- HuggingFace for providing the transformer models
- Facebook Research for FAISS
- Streamlit team for the web framework

## Contact

For any questions or issues, please open an issue in the repository.