import os
import warnings
import streamlit.web.bootstrap
from streamlit.web.server import Server
from streamlit.runtime import Runtime

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

# Patch Streamlit's file watcher to avoid torch class registration issue
def _get_paths(module):
    try:
        return list(module.__path__)
    except (AttributeError, RuntimeError):
        return []
        
streamlit.web.bootstrap.get_module_paths = _get_paths

# Import remaining dependencies after environment setup
import torch
import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from agent import QAAgent

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state['qa_system'] = None
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ''

# Configure Streamlit
st.set_page_config(
    page_title="AI Knowledge Assistant",
    layout="centered"
)

def initialize_qa_system():
    """Initialize the QA system"""
    try:
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        
        with st.spinner("Processing documents..."):
            documents = doc_processor.process_documents()
            vector_store.create_vector_store(documents)
        
        return QAAgent(vector_store)
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

def set_question(question: str):
    """Set the question in session state"""
    st.session_state['current_question'] = question

def main():
    st.title("AI Knowledge Assistant")
    st.write("Ask me anything about the documents in the knowledge base!")

    # Sidebar for debug information
    with st.sidebar:
        st.subheader("Debug Information")
        if st.session_state.get('qa_system'):
            st.info("System Status: Ready")
            if st.button("Force Retrain"):
                st.session_state['qa_system'] = None
                st.rerun()
        else:
            st.warning("System Status: Initializing")

    # Initialize QA system if not already done
    if st.session_state['qa_system'] is None:
        with st.spinner("Loading AI model and knowledge base..."):
            st.session_state['qa_system'] = initialize_qa_system()

    # Create the query input if system is initialized
    if st.session_state['qa_system']:
        # Use the session state value as default for the text input
        query = st.text_input("Enter your question:", value=st.session_state['current_question'])
        
        # Process the query if it exists
        if query:
            try:
                with st.spinner("Processing your question..."):
                    with torch.inference_mode():
                        result = st.session_state['qa_system'].process_query(query)
                    
                    # Display which tool was used
                    st.info(f"Tool Used: {result['tool_used']}")
                    
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    
                    # Clear the question after processing
                    st.session_state['current_question'] = ''
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

        # Sample Questions Section
        st.markdown("---")  # Divider
        st.subheader("Try these sample questions:")
          # Define two sample questions
        sample_questions = [
            "Calculate 10+12",
            "Define algorithm"
        ]

        # Display questions vertically
        for question in sample_questions:
            if st.button(question, key=f"btn_{question}"):
                set_question(question)
                st.rerun()

if __name__ == "__main__":
    with torch.inference_mode():
        main()