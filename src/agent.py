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

    def _word_to_number(self, word: str) -> str:
        """Convert word numbers to digits"""
        word = word.lower().strip()
        number_mapping = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'plus': '+', 'minus': '-', 'times': '*', 'multiplied by': '*',
            'divided by': '/', 'over': '/'
        }
        return number_mapping.get(word, word)

    def _calculate(self, expression: str) -> str:
        """Handle mathematical calculations"""
        try:
            # Extract numbers and operator from the expression
            expression = expression.replace('calculate', '').strip()
            
            # Convert words to numbers/operators
            words = expression.lower().split()
            converted = []
            i = 0
            while i < len(words):
                # Handle two-word operators
                if i < len(words) - 1 and f"{words[i]} {words[i+1]}" in ["multiplied by", "divided by"]:
                    converted.append(self._word_to_number(f"{words[i]} {words[i+1]}"))
                    i += 2
                else:
                    converted.append(self._word_to_number(words[i]))
                    i += 1
            
            # Join and clean the expression
            expression = ' '.join(converted)
            # Remove any remaining text and clean up spacing
            expression = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', expression)
            expression = re.sub(r'\s+', '', expression)
            
            # Validate the expression
            if not re.match(r'^[\d\+\-\*\/\(\)\.]+$', expression):
                return "Invalid calculation expression. Please use numbers and basic operators (+, -, *, /)."
            
            result = eval(expression)
            # Format the result to handle floating point precision
            if isinstance(result, float):
                result = round(result, 4)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Sorry, I couldn't perform that calculation. Please check your expression."

    def _define(self, word: str) -> str:
        """Look up word definitions using a dictionary API"""
        try:
            # Remove dictionary-related keywords and clean the word
            word = word.lower()
            dictionary_keywords = ['define', 'explain', 'what is', 'tell me about', 'describe', 'meaning of', 'define the term', 'define the word']
            for keyword in dictionary_keywords:
                word = word.replace(keyword, '').strip()
            
            # Remove any remaining articles or common words
            word = re.sub(r'^(a |an |the |what |is |are )', '', word).strip()
            
            # Make the API call
            response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
            if response.status_code == 200:
                data = response.json()[0]
                meaning = data['meanings'][0]['definitions'][0]['definition']
                return f"Definition of {word}: {meaning}"
            return f"Sorry, I couldn't find a definition for '{word}'"
        except Exception as e:
            return f"Sorry, I couldn't look up that word. Please try again."

    def _preprocess_query(self, query: str) -> tuple:
        """Preprocess the query and determine the appropriate tool"""
        query = query.lower().strip()
        
        # Check for calculation
        if "calculate" in query:
            return query, "calculator"
            
        # Check for definition-like queries
        definition_patterns = [
            r'^define\s+',
            r'^explain\s+',
            r'^what\s+is\s+',
            r'^what\s+are\s+',
            r'^tell\s+me\s+about\s+',
            r'^describe\s+',
            r'^meaning\s+of\s+'
        ]
        
        for pattern in definition_patterns:
            if re.match(pattern, query):
                return query, "dictionary"
        
        # For general queries, add question mark if missing
        if not query.endswith("?"):
            query += "?"
        return query, "search"

    def process_query(self, query: str) -> dict:
        """Process user query and generate response"""
        try:
            # Preprocess the query and determine tool
            processed_query, tool = self._preprocess_query(query)
            
            # Route to appropriate tool
            if tool == "calculator":
                return {
                    "tool_used": "Calculator",
                    "answer": self._calculate(processed_query)
                }
            elif tool == "dictionary":
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