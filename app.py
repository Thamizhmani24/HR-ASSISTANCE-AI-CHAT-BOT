import streamlit as st
import os
import tempfile

# Automatically sync Streamlit Cloud Secrets to os.environ if present
try:
    if hasattr(st, "secrets"):
        for key, val in st.secrets.items():
            if isinstance(val, str):
                os.environ[key] = val
except Exception:
    pass

from QueryProcessor import process_user_query
from dataprocessor import run as run_data_processor


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="RAG HR Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
    st.title("💼 HR Assistant")
    st.caption("AI-Powered RAG System for HR Queries")
    
    st.divider()
    
    st.subheader("⚙️ System Status")
    pinecone_index = os.getenv("PINECONE_INDEX_NAME", "hr-assistant")
    llm_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    
    st.success(f"**Vector DB**: `{pinecone_index}`")
    st.info(f"**LLM Model**: `{llm_model}`")
    st.info(f"**Embedding**: `{embedding_model}`")
    
    st.divider()
    
    st.subheader("📄 Upload & Re-index Document")
    uploaded_file = st.file_uploader("Upload an HR Policy PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Update Vector DB", use_container_width=True):
            with st.spinner("Processing PDF and storing vectors in Pinecone..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    chunks_count = run_data_processor(tmp_path)
                    os.remove(tmp_path)
                    st.success(f"Success! Processed {chunks_count} chunks into Pinecone.")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
    else:
        if st.button("Re-index Default HRPolicy.pdf", use_container_width=True):
            with st.spinner("Processing default HRPolicy.pdf..."):
                try:
                    chunks_count = run_data_processor("./resources/HRPolicy.pdf")
                    st.success(f"Successfully indexed {chunks_count} chunks!")
                except Exception as e:
                    st.error(f"Failed to index default PDF: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.markdown('<div class="main-header">RAG HR Policy Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about company HR policies, leave rules, work hours, benefits, and guidelines.</div>', unsafe_allow_html=True)

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI HR Assistant. How can I help you regarding company policies today?",
            "sources": []
        }
    ]

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 View Retrieved Context (Sources)", expanded=False):
                for idx, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.markdown(f'<div class="source-box">{src}</div>', unsafe_allow_html=True)

# Handle User Input
if prompt := st.chat_input("Ask a question about HR Policy..."):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query using RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching HR Policy vector database & generating answer..."):
            try:
                answer, sources = process_user_query(prompt)
                st.markdown(answer)
                
                if sources:
                    with st.expander("📄 View Retrieved Context (Sources)", expanded=False):
                        for idx, src in enumerate(sources, start=1):
                            st.markdown(f"**Chunk {idx}:**")
                            st.markdown(f'<div class="source-box">{src}</div>', unsafe_allow_html=True)
                            
                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                error_msg = f"An error occurred while processing your request: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
