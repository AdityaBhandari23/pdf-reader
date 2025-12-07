import os
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mangum import Mangum
from pydantic import BaseModel
import time

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate API keys are loaded
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Constants
PINECONE_INDEX_NAME = "pdf-chat"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Initialize LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


class QueryRequest(BaseModel):
    question: str
    session_id: str


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload a PDF, process it in BATCHES with delays to avoid 429 Errors.
    Stores documents in Pinecone with namespace=session_id for user isolation.
    """
    temp_file_path = None
    try:
        # 1. Save uploaded file to temp directory
        temp_dir = Path(tempfile.gettempdir())
        temp_file_path = temp_dir / file.filename
        
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # 2. Load and Split PDF
        loader = PyPDFLoader(str(temp_file_path))
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"Total chunks to process: {len(chunks)} for session_id: {session_id}")

        # --- BATCHING LOGIC WITH NAMESPACE ISOLATION ---
        batch_size = 5  # Process only 5 chunks at a time
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(f"Processing batch {i} to {i + batch_size}...")
            
            # Upload just this small batch with namespace=session_id for user isolation
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                pinecone_api_key=PINECONE_API_KEY,
                namespace=session_id  # CRITICAL: Isolate user's PDF data
            )
            
            # CRITICAL: Sleep for 2 seconds to let Google's API "cool down"
            time.sleep(2) 
        # --- BATCHING LOGIC ENDS HERE ---
        
        # 3. Cleanup
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "chunks_processed": len(chunks),
                "message": "PDF processed successfully in batches.",
                "session_id": session_id
            }
        )
    
    except Exception as e:
        # Cleanup in case of error
        if temp_file_path and temp_file_path.exists():
             try:
                 temp_file_path.unlink()
             except:
                 pass
        raise HTTPException(status_code=500, detail=str(e))
##previous
# @app.post("/upload-pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     """
#     Upload a PDF file, process it, and store it in Pinecone vector database.
#     """
#     temp_file_path = None
#     try:
#         # Validate file type
#         if not file.filename.endswith('.pdf'):
#             raise HTTPException(
#                 status_code=400,
#                 detail="File must be a PDF"
#             )
        
#         # Save uploaded file to /tmp/ directory (AWS Lambda compatible)
#         temp_dir = Path("/tmp")
#         temp_dir.mkdir(exist_ok=True)
#         temp_file_path = temp_dir / file.filename
        
#         # Write the uploaded file to temporary location
#         with open(temp_file_path, "wb") as temp_file:
#             content = await file.read()
#             temp_file.write(content)
        
#         # Load PDF using PyPDFLoader
#         loader = PyPDFLoader(str(temp_file_path))
#         documents = loader.load()
        
#         # Split text using RecursiveCharacterTextSplitter
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_documents(documents)
        
#         # Create Pinecone vector store using .from_documents()
#         vector_store = PineconeVectorStore.from_documents(
#             documents=chunks,
#             embedding=embeddings,
#             index_name=PINECONE_INDEX_NAME,
#             pinecone_api_key=PINECONE_API_KEY
#         )
        
#         # Delete temporary file
#         if temp_file_path and temp_file_path.exists():
#             temp_file_path.unlink()
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "status": "success",
#                 "chunks_processed": len(chunks)
#             }
#         )
    
#     except Exception as e:
#         # Clean up temporary file in case of error
#         if temp_file_path and temp_file_path.exists():
#             try:
#                 temp_file_path.unlink()
#             except:
#                 pass
        
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing PDF: {str(e)}"
#         )


@app.post("/chat")
async def chat(query: QueryRequest):
    """
    Query the RAG system with a question and get an answer with sources.
    Uses DynamoDB for chat history and namespace isolation for user-specific document search.
    """
    try:
        # Initialize DynamoDB chat message history for this session
        history = DynamoDBChatMessageHistory(
            table_name="pdf-chat-history",
            session_id=query.session_id
        )
        
        # Get existing conversation history
        history_messages = history.messages
        
        # Step 1: Contextualize the question based on chat history
        rewritten_question = query.question
        if history_messages:
            # Build history context for rewriting
            history_context = "\n".join([
                f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
                for msg in history_messages[-6:]  # Use last 6 messages for context
            ])
            
            rewrite_prompt = f"""Based on the following conversation history, rewrite the user's question to be standalone and explicit.
If the question is a follow-up (e.g., "What is his email?"), make it explicit (e.g., "What is Aditya's email?").

Conversation History:
{history_context}

User's Current Question: {query.question}

Rewritten Question (make it explicit and standalone):"""
            
            rewrite_response = llm.invoke(rewrite_prompt)
            rewritten_question = rewrite_response.content if hasattr(rewrite_response, 'content') else str(rewrite_response)
            rewritten_question = rewritten_question.strip()
        
        # Step 2: Initialize Pinecone vector store with namespace isolation
        # CRITICAL: namespace=session_id ensures we ONLY search this user's documents
        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY,
            namespace=query.session_id  # Isolated search for this user
        )
        
        # Step 3: Perform similarity search using the rewritten question
        docs = vector_store.similarity_search(
            rewritten_question,
            k=TOP_K_RESULTS
        )
        
        # Extract source texts
        sources = [doc.page_content for doc in docs]
        
        # Step 4: Build context from retrieved documents
        context = "\n\n".join([f"Context {i+1}: {source}" for i, source in enumerate(sources)])
        
        # Step 5: Build conversation history for the final prompt
        history_text = ""
        if history_messages:
            history_text = "\n\nPrevious Conversation:\n" + "\n".join([
                f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
                for msg in history_messages[-10:]  # Include last 10 messages
            ])
        
        # Step 6: Create final prompt with original question, context, and history
        prompt = f"""Context from Documents:
{context}
{history_text}

Current Question: {query.question}

Please answer the question based on the provided context and conversation history. If the context doesn't contain enough information to answer the question, please say so."""
        
        # Step 7: Get response from LLM
        response = llm.invoke(prompt)
        
        # Extract answer text from response
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Step 8: Save the interaction to DynamoDB
        history.add_user_message(query.question)
        history.add_ai_message(answer)
        
        return JSONResponse(
            status_code=200,
            content={
                "answer": answer,
                "sources": sources
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat query: {str(e)}"
        )


@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Retrieve chat history for a given session_id from DynamoDB.
    Returns a list of messages with role and content.
    """
    try:
        # Initialize DynamoDB chat message history for this session
        history = DynamoDBChatMessageHistory(
            table_name="pdf-chat-history",
            session_id=session_id
        )
        
        # Get all messages from history
        history_messages = history.messages
        
        # Convert LangChain messages to JSON format
        messages = []
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                messages.append({
                    "role": "user",
                    "content": msg.content,
                    "sources": []  # Sources are not stored in history
                })
            elif isinstance(msg, AIMessage):
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "sources": []  # Sources are not stored in history
                })
        
        return JSONResponse(
            status_code=200,
            content=messages
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "PDF Chat RAG API is running"}


# Wrap the FastAPI app with Mangum for AWS Lambda compatibility
handler = Mangum(app)

