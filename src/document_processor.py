from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        # Using RecursiveCharacterTextSplitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=500,  # Smaller chunks for better context preservation
            chunk_overlap=50,  # Reduced overlap to prevent redundancy
            length_function=len
        )
    
    def load_documents(self) -> List[Document]:
        """Load documents from the data directory"""
        documents = []
        print("Loading documents from:", self.data_dir)
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            if text.strip():  # Only add non-empty documents
                                documents.append(Document(
                                    page_content=text,
                                    metadata={
                                        "source": file_path,
                                        "filename": file
                                    }
                                ))
                                print(f"Successfully loaded {file} ({len(text)} characters)")
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def process_documents(self) -> List[Document]:
        """Load and split documents into chunks"""
        documents = self.load_documents()
        if not documents:
            print("No documents were loaded!")
            return []
        
        text_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            print(f"Split {doc.metadata['filename']} into {len(chunks)} chunks")
            text_chunks.extend([
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_size": len(chunk)
                    }
                ) for chunk in chunks
            ])
        
        print(f"Total chunks created: {len(text_chunks)}")
        return text_chunks