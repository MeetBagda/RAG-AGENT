import unittest
from unittest.mock import MagicMock, patch
from src.agent import QAAgent
from langchain_core.documents import Document

class TestQAAgent(unittest.TestCase):
    def setUp(self):
        # Create a mock vector store
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.similarity_search.return_value = [
            Document(
                page_content="Healthcare and Finance are our main industries",
                metadata={"source": "test.txt"}
            )
        ]
        
        # Initialize agent with mock vector store
        self.agent = QAAgent(self.mock_vector_store)
    
    def test_process_query(self):
        # Test normal query processing
        result = self.agent.process_query("What industries do we serve?")
        self.assertIsInstance(result, dict)
        self.assertIn("tool_used", result)
        self.assertIn("answer", result)
        self.assertEqual(result["tool_used"], "Search")
        
    def test_empty_result_handling(self):
        # Test handling when no relevant documents found
        self.mock_vector_store.similarity_search.return_value = []
        result = self.agent.process_query("What is the meaning of life?")
        self.assertEqual(
            result["answer"],
            "I couldn't find any relevant information to answer your question."
        )
    
    def test_error_handling(self):
        # Test error handling
        self.mock_vector_store.similarity_search.side_effect = Exception("Test error")
        result = self.agent.process_query("Test query")
        self.assertEqual(result["tool_used"], "Error")
        self.assertTrue(result["answer"].startswith("An error occurred"))

if __name__ == "__main__":
    unittest.main()