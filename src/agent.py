import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from langchain_core.prompts import PromptTemplate
import re
import requests

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

    def _calculate(self, expression: str) -> str:
        """Handle mathematical calculations"""
        try:
            # Extract numbers and operator from the expression
            expression = expression.replace('calculate', '').strip()
            # Clean and validate the expression
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
                return "Invalid calculation expression. Please use only numbers and basic operators (+, -, *, /)."
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Sorry, I couldn't perform that calculation. Please check your expression."

    def _define(self, word: str) -> str:
        """Look up word definitions using a dictionary API"""
        try:
            # Using Free Dictionary API
            word = word.replace('define', '').strip()
            response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
            if response.status_code == 200:
                data = response.json()[0]
                meaning = data['meanings'][0]['definitions'][0]['definition']
                return f"Definition of {word}: {meaning}"
            return f"Sorry, I couldn't find a definition for '{word}'"
        except Exception as e:
            return f"Sorry, I couldn't look up that word. Please try again."

    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query to handle different phrasings"""
        query = query.lower().strip()
        if query.startswith("tell me"):
            query = query.replace("tell me", "what is", 1)
        if not query.endswith("?") and not any(x in query for x in ["calculate", "define"]):
            query += "?"
        return query

    def process_query(self, query: str) -> dict:
        """Process user query and generate response"""
        try:
            processed_query = self._preprocess_query(query)
            
            # Route to appropriate tool based on keywords
            if "calculate" in processed_query:
                return {
                    "tool_used": "Calculator",
                    "answer": self._calculate(processed_query)
                }
            elif "define" in processed_query:
                return {
                    "tool_used": "Dictionary",
                    "answer": self._define(processed_query)
                }
            
            # Default RAG pipeline for other queries
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
                    temperature=0.7,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
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