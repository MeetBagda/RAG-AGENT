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
        ).to("cpu")  # Explicitly use CPU
        
        self.vector_store = vector_store
        self.prompt = PromptTemplate.from_template(
            """Based on the following context, answer the question accurately and completely. 
            If the answer is a list, make sure to include all items. 
            If you cannot find the answer in the context, say so.
            
            Context: {context}
            Question: {question}
            
            Provide a complete answer:"""
        )

    def process_query(self, query: str) -> dict:
        """Process user query and generate response"""
        try:
            # Get relevant documents with increased k for better context
            docs = self.vector_store.similarity_search(query, k=5)
            if not docs:
                return {
                    "tool_used": "Search",
                    "answer": "I couldn't find any relevant information to answer your question."
                }
            
            # Prepare context and prompt
            context = "\n".join([doc.page_content for doc in docs])
            prompt = self.prompt.format(context=context, question=query)
            
            # Add task prefix to help model understand the task better
            input_text = f"Answer the question: {prompt}"
            
            # Generate answer with adjusted parameters
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,  # Increased for more complete answers
                    min_length=20,   # Ensure answers aren't too short
                    num_beams=5,     # Increased for better search
                    temperature=0.3,  # Reduced for more focused answers
                    no_repeat_ngram_size=3,
                    length_penalty=1.0  # Encourage slightly longer answers
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