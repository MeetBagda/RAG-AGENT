import unittest
import os
import shutil
from src.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary test data directory
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create a test document
        with open(os.path.join(self.test_data_dir, "test.txt"), "w", encoding="utf-8") as f:
            f.write("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
        
        self.processor = DocumentProcessor(data_dir=self.test_data_dir)
    
    def tearDown(self):
        # Clean up test directory
        shutil.rmtree(self.test_data_dir)
    
    def test_load_documents(self):
        documents = self.processor.load_documents()
        self.assertEqual(len(documents), 1)
        self.assertTrue("test.txt" in documents[0].metadata["filename"])
    
    def test_process_documents(self):
        chunks = self.processor.process_documents()
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(isinstance(chunk.page_content, str) for chunk in chunks))
        self.assertTrue(all("chunk_size" in chunk.metadata for chunk in chunks))

if __name__ == "__main__":
    unittest.main()