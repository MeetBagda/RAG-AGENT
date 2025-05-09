import unittest
import os
import shutil
from langchain_core.documents import Document
from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.vector_store = VectorStore()
        self.test_docs = [
            Document(page_content="Test document one", metadata={"source": "test1.txt"}),
            Document(page_content="Test document two", metadata={"source": "test2.txt"}),
        ]
    
    def tearDown(self):
        # Clean up vector store if it exists
        if os.path.exists(self.vector_store.store_path):
            shutil.rmtree(self.vector_store.store_path)
    
    def test_create_vector_store(self):
        result = self.vector_store.create_vector_store(self.test_docs)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(self.vector_store.store_path))
    
    def test_similarity_search(self):
        self.vector_store.create_vector_store(self.test_docs)
        results = self.vector_store.similarity_search("test document", k=1)
        self.assertEqual(len(results), 1)
        self.assertTrue(isinstance(results[0], Document))

if __name__ == "__main__":
    unittest.main()