import streamlit as st
import requests
from typing import List, Dict
import time

# Page configuration
st.set_page_config(
    page_title="AI RAG Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Backend API URL
BACKEND_URL = "http://127.0.0.1:8000"


def login_screen():
    """Display the login screen"""
    # Center the login card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Login card
        with st.container():
            st.markdown("""
                <div style='text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h1 style='color: #1f77b4; margin-bottom: 1rem;'>Welcome to AI RAG Chat</h1>
                    <p style='color: #666; margin-bottom: 2rem;'>Upload PDFs and chat with your documents using AI</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Email input
            email = st.text_input(
                "Enter your Email to Start",
                key="email_input",
                placeholder="your.email@example.com",
                label_visibility="collapsed"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Start chatting button
            if st.button("Start Chatting", type="primary", use_container_width=True):
                if email and email.strip():
                    st.session_state.session_id = email.strip()
                    st.session_state.logged_in = True
                    st.session_state.messages = []  # Clear any previous messages
                    st.rerun()
                else:
                    st.error("Please enter a valid email address")


def chat_interface():
    """Display the main chat interface"""
    
    # Sidebar
    with st.sidebar:
        st.title("💬 AI RAG Chat")
        st.divider()
        
        # Display logged in user
        st.markdown(f"**Logged in as:**\n\n`{st.session_state.session_id}`")
        
        st.divider()
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.last_uploaded_file = None
            st.rerun()
        
        st.divider()
        
        # File upload section
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="collapsed",
            key="pdf_uploader"
        )
        
        # Only process if file is newly uploaded (different from last one)
        if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
            try:
                with st.spinner("Uploading and processing PDF..."):
                    # Prepare the file and form data
                    files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
                    data = {"session_id": st.session_state.session_id}
                    
                    # Make API call
                    response = requests.post(
                        f"{BACKEND_URL}/upload-pdf",
                        files=files,
                        data=data,
                        timeout=300  # 5 minutes timeout for large files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ PDF uploaded successfully! ({result.get('chunks_processed', 0)} chunks processed)")
                        st.session_state.last_uploaded_file = uploaded_file.name
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                        
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Please ensure the server is running at http://127.0.0.1:8000")
            except requests.exceptions.Timeout:
                st.error("❌ Upload timeout. The file might be too large.")
            except Exception as e:
                st.error(f"❌ Error uploading file: {str(e)}")
        
        st.divider()
        st.markdown("""
        <div style='font-size: 0.8rem; color: #666; text-align: center;'>
            <p>💡 Tip: Upload a PDF first, then ask questions about it!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main chat area
    st.title("💬 Chat with Your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        sources = message.get("sources", [])
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Show sources if available (for assistant messages)
            if role == "assistant" and sources:
                with st.expander("📚 View Source Documents"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:500] + "..." if len(source) > 500 else source)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to UI immediately
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Make API call to chat endpoint
                    response = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No answer provided.")
                        sources = result.get("sources", [])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Show sources in expander
                        if sources:
                            with st.expander("📚 View Source Documents"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(source[:500] + "..." if len(source) > 500 else source)
                                    st.divider()
                        
                        # Add assistant message to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = f"❌ Error: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })
                        
                except requests.exceptions.ConnectionError:
                    error_msg = "❌ Cannot connect to backend. Please ensure the server is running at http://127.0.0.1:8000"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
                except requests.exceptions.Timeout:
                    error_msg = "❌ Request timeout. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
                except Exception as e:
                    error_msg = f"❌ Unexpected error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
        
        # Rerun to update the UI
        st.rerun()


def main():
    """Main application logic"""
    # Check if user is logged in
    if not st.session_state.logged_in:
        login_screen()
    else:
        chat_interface()


if __name__ == "__main__":
    main()

