## AI RAG Chat (FastAPI + Streamlit)

A **production-style Retrieval-Augmented Generation (RAG) SaaS demo** for chatting with your own PDFs.

- **Backend**: `FastAPI` + `LangChain` + `Pinecone` + `Google Gemini`
- **Frontend**: `Streamlit` chat UI with login-style email and multi-user isolation
- **Memory**: Per-user conversational history stored in **AWS DynamoDB**
- **Serverless Ready**: FastAPI app wrapped with `Mangum` for AWS Lambda

---

## 1. Features

- **PDF Upload & Indexing**
  - Upload PDFs from the Streamlit UI
  - Text extraction via `PyPDFLoader`
  - Chunking with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`)
  - Stored in **Pinecone** under a **namespace = session_id (user email)**
  - Batched uploads (5 chunks at a time + 2s sleep) to avoid 429 rate limits

- **RAG Chat**
  - Uses `GoogleGenerativeAIEmbeddings` (`models/embedding-001`) for vectorization
  - Uses `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) as the LLM
  - Retrieves top-3 relevant chunks from Pinecone
  - Builds a prompt with **document context + conversation history + current question**

- **Multi-User Isolation**
  - `session_id` = user email
  - Pinecone namespace = `session_id` → users can only search their own documents
  - Chat history per user is stored in DynamoDB keyed by `SessionId`

- **Conversational Memory (DynamoDB)**
  - History stored using `DynamoDBChatMessageHistory`
  - On every question:
    - History is loaded from DynamoDB
    - The question is **rewritten** to be standalone using the LLM (handles “his”, “her”, “that” references)
    - The rewritten question is used for vector search
    - The original question + history + context are used to generate the final answer
  - History is persisted across frontend restarts

- **Chat History API**
  - `GET /chat-history/{session_id}` returns a list of messages:
    - `{"role": "user" | "assistant", "content": "...", "sources": []}`
  - `HumanMessage` → `user`, `AIMessage` → `assistant`

- **Frontend UX (Streamlit)**
  - Login-like screen asking for **Email**
  - Sidebar with:
    - `Logged in as: [email]`
    - Logout button
    - PDF uploader
  - Main area:
    - Chat-style interface using `st.chat_message("user")` / `st.chat_message("assistant")`
    - `st.chat_input` for questions
    - `View Source Documents` expanders showing the RAG context

---

## 2. Project Structure

- `main.py` – FastAPI backend (RAG, Pinecone, Gemini, DynamoDB, Mangum handler)
- `frontend.py` – Streamlit SaaS-style frontend UI
- `requirements.txt` – Python dependencies
- `readme.md` – Project documentation (this file)
- `.env` – **Your real secrets** (NOT committed)
- `.env.example` – Example env file with dummy values

---

## 3. Backend (FastAPI) Overview

### 3.1 Tech Stack

- **FastAPI** for the HTTP API
- **LangChain** components:
  - `PyPDFLoader`
  - `RecursiveCharacterTextSplitter`
  - `GoogleGenerativeAIEmbeddings`
  - `ChatGoogleGenerativeAI`
  - `DynamoDBChatMessageHistory`
- `PineconeVectorStore` from `langchain_pinecone`
- `Mangum` to wrap FastAPI for AWS Lambda

### 3.2 Environment Variables Used

`main.py` loads all configuration from `.env` via `python-dotenv`:

- **Google / Gemini**
  - `GOOGLE_API_KEY`

- **Pinecone**
  - `PINECONE_API_KEY`

- **AWS / DynamoDB**
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`

These must be **set in the environment** or in a local `.env` file (see `.env.example`).

### 3.3 Key Endpoints

- **Health Check**
  - `GET /`
  - Returns a simple JSON message to confirm the API is running.

- **Upload PDF**
  - `POST /upload-pdf`
  - **Body (multipart/form-data)**:
    - `file`: PDF file (UploadFile)
    - `session_id`: string (user email from frontend)
  - **Flow**:
    1. Save file to temp dir (`tempfile.gettempdir()`)
    2. Load with `PyPDFLoader`
    3. Split into chunks (`chunk_size=1000`, `chunk_overlap=200`)
    4. In batches of 5 chunks:
       - Call `PineconeVectorStore.from_documents(..., namespace=session_id)`
       - Sleep 2 seconds to prevent Gemini rate limits
    5. Delete the temp file
  - **Response (200)**:
    - `{"status": "success", "chunks_processed": <int>, "message": "...", "session_id": "..."}`.

- **Chat (RAG + Memory)**
  - `POST /chat`
  - **Body (JSON)** `QueryRequest`:
    - `question: str`
    - `session_id: str`
  - **Flow**:
    1. Initialize `DynamoDBChatMessageHistory(table_name="pdf-chat-history", session_id=session_id)`
    2. Load `history.messages`
    3. Build a history context string (last few messages)
    4. Ask Gemini to **rewrite** the question to be explicit and standalone
    5. Use rewritten question to query Pinecone via `PineconeVectorStore(..., namespace=session_id)`
    6. Build a prompt combining:
       - Retrieved document context
       - Previous conversation
       - Original question
    7. Call `llm.invoke(prompt)`
    8. Save `user` and `assistant` messages back into DynamoDB
    9. Return `{"answer": "...", "sources": ["chunk1", "chunk2", ...]}`.

- **Chat History**
  - `GET /chat-history/{session_id}`
  - **Returns**: full ordered history for that user as a list of messages:
    - `[{ "role": "user", "content": "...", "sources": [] }, { "role": "assistant", "content": "...", "sources": [] }, ...]`.

---

## 4. Frontend (Streamlit) Overview

### 4.1 Login Flow (Pseudo-SaaS)

- Uses `st.session_state`:
  - `logged_in: bool`
  - `session_id: str` (the email)
  - `messages: List[Dict]` for UI rendering
  - `last_uploaded_file: Optional[str]` to prevent duplicate uploads

- **Screen 1 – Login**
  - Centered card with:
    - Title: **"Welcome to AI RAG Chat"**
    - Description: short subtitle
    - Email input: `st.text_input`
    - Button: `Start Chatting`
  - On click:
    - Validate email
    - Set `session_id = email`
    - Set `logged_in = True`
    - Call `GET /chat-history/{email}` to preload past messages
    - Save result into `st.session_state.messages`
    - `st.rerun()` into chat screen

### 4.2 Chat Screen

- **Sidebar**
  - Shows `Logged in as: [email]`
  - **Logout button** resets all session state and reloads login screen
  - PDF uploader:
    - Uses `requests.post` to send `file` + `session_id` to `/upload-pdf`
    - Shows success / error using `st.success` / `st.error`

- **Main Chat Area**
  - Renders `st.session_state.messages` as a chat timeline:
    - `role == "user"` → `st.chat_message("user")`
    - `role == "assistant"` → `st.chat_message("assistant")`
  - Each assistant message can show sources inside `st.expander("View Source Documents")`
  - `st.chat_input` sends the question:
    - Immediately appends a `user` message to `st.session_state.messages`
    - Calls `POST /chat` with `{"question": prompt, "session_id": email}`
    - Appends an `assistant` message with `answer` and `sources`

---

## 5. Running the Project Locally

### 5.1 Create and Activate Virtual Environment (optional)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 5.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 5.3 Configure Environment Variables

1. Copy `.env.example` to `.env`:

```bash
cp .env.example .env   # macOS / Linux
# OR (Windows PowerShell)
copy .env.example .env
```

2. Replace the **dummy values** with your real keys.

### 5.4 Start the Backend (FastAPI)

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 5.5 Start the Frontend (Streamlit)

```bash
streamlit run frontend.py
```

The UI will be available at `http://localhost:8501`.

---

## 6. Environment Configuration (`.env.example`)

Below is an example of what your `.env` file should look like. **Do NOT commit your real `.env` file to Git.**

```env
# Google Gemini (Generative AI) API key
GOOGLE_API_KEY=AIzaSyDUMMY-KEY-FOR-EXAMPLE-ONLY-123456789

# Pinecone API key (Vector database)
PINECONE_API_KEY=pcsk_dummy_pinecone_key_123456789

# AWS credentials for DynamoDB chat history
AWS_ACCESS_KEY_ID=AKIAFAKEACCESSKEY1234
AWS_SECRET_ACCESS_KEY=fakeSecretKeyForExampleOnlyDontUseInProd123456
AWS_DEFAULT_REGION=us-east-1
```

> **Important**: The keys above are **fake** and only for documentation. Replace them with your own valid keys before running the app.

---

## 7. Deployment Notes

- **Backend**
  - FastAPI app is wrapped with `Mangum(app)` → ready for AWS Lambda + API Gateway.
  - Ensure environment variables are configured in the Lambda environment.
  - Pinecone index `pdf-chat` must exist with the correct dimensions for `models/embedding-001`.

- **Frontend**
  - Streamlit can be deployed on Streamlit Cloud, EC2, or any container platform.
  - Update `BACKEND_URL` in `frontend.py` to point to your deployed FastAPI / API Gateway URL.

- **Persistence & Limits**
  - DynamoDB table: `pdf-chat-history` with partition key `SessionId` (string).
  - Be mindful of Pinecone storage / query costs and Gemini rate limits.

---

## 8. Troubleshooting

- **Cannot connect to backend**
  - Check that FastAPI is running on `http://127.0.0.1:8000` (or update `BACKEND_URL`).

- **401 / auth errors from APIs**
  - Verify that all keys in `.env` are correct and loaded.

- **No answers / empty context**
  - Ensure you have uploaded at least one PDF for that email (`session_id`).
  - Check Pinecone index and namespaces for data.

- **History not loading after restart**
  - Confirm DynamoDB table `pdf-chat-history` exists and IAM permissions are correct.

---

Happy hacking with your AI RAG Chat SaaS demo!


