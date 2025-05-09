import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from langchain_core.prompts import PromptTemplate

class QAAgent:
    def __init__(self, vector_store):
        # Configure PyTorch
        torch.set_grad_enabled(False)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Initialize model and tokenizer
        model_name = "google/flan-t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to("cpu")
        
        self.vector_store = vector_store
        # Updated prompt template to handle more question variations
        self.prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant. Using the following context, answer the question.
            Consider different ways the question might be phrased and look for relevant information.
            If you find the answer, provide it in a clear and natural way.
            If the answer is a list, include all items.
            If you cannot find the answer in the context, say so.
            
            Context: {context}
            Question: {question}
            
            Answer in a natural, conversational way:"""
        )

    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query to handle different phrasings"""
        # Convert commands to questions
        query = query.lower().strip()
        if query.startswith("tell me"):
            query = query.replace("tell me", "what is", 1)
        if not query.endswith("?"):
            query += "?"
        return query

    def process_query(self, query: str) -> dict:
        """Process user query and generate response"""
        try:
            # Preprocess the query for better matching
            processed_query = self._preprocess_query(query)
            
            # Get relevant documents with increased k for better context
            docs = self.vector_store.similarity_search(processed_query, k=5)
            if not docs:
                return {
                    "tool_used": "Search",
                    "answer": "I couldn't find any relevant information to answer your question."
                }
            
            # Prepare context and prompt
            context = "\n".join([doc.page_content for doc in docs])
            prompt = self.prompt.format(context=context, question=processed_query)
            
            # Add task prefix for better model understanding
            input_text = f"Answer this: {prompt}"
            
            # Generate answer with optimized parameters
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    min_length=20,
                    num_beams=5,
                    temperature=0.7,  # Increased for more natural responses
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    do_sample=True,   # Enable sampling for more natural responses
                    top_k=50,         # Limit vocabulary for more focused responses
                    top_p=0.95        # Use nucleus sampling for better text quality
                )
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "tool_used": "Search",
                "answer": answer.strip()
            }
        except Exception as e:
            return {
                "tool_used": "Error",
                "answer": f"An error occurred: {str(e)}"
            }