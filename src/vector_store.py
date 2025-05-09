from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class VectorStore:
    def __init__(self):
        # Use the smallest but effective model for embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.store_path = "vector_store"

    def create_vector_store(self, documents):
        """Create a new vector store from documents"""
        try:
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            self._save_vector_store()
            return self.vector_store
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None

    def load_vector_store(self):
        """Load an existing vector store"""
        try:
            if os.path.exists(self.store_path):
                self.vector_store = FAISS.load_local(
                    self.store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # We trust our local files
                )
                return self.vector_store
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
        return None

    def _save_vector_store(self):
        """Save the vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(self.store_path)

    def similarity_search(self, query, k=3):
        """Search for similar documents"""
        try:
            if not self.vector_store:
                self.load_vector_store()
            if self.vector_store:
                return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
        return []