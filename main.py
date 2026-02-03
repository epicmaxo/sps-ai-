from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select
from database import engine, create_db_and_tables, Conversation, Message, get_session
from typing import List, Optional
from pydantic import BaseModel
import os
import openai
from datetime import datetime

# Initialize App
app = FastAPI(title="SPS AI Minimal")

# --- UI Endpoint ---
# --- UI Endpoint ---
@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_ui():
    print("GET /chat-ui requested")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "templates", "chat_iframe.html")
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        print(f"Error serving chat UI: {e}")
        return HTMLResponse(content=f"<h1>Error loading UI</h1><p>{e}</p>", status_code=500)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup
@app.on_event("startup")
def on_startup():
    print("--- RELAUNCHING WITH ROBUST STARTUP FIXES ---")
    print("--- Starting SPS AI Service ---")
    try:
        create_db_and_tables()
        print("--- Database Connected & Tables Created ---")
    except Exception as e:
        print(f"CRITICAL ERROR: Database connection failed: {e}")

# OpenAI Client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY not set in environment. Chat endpoints will fail.")
    openai_client = None
else:
    try:
        openai_client = openai.OpenAI(api_key=api_key)
        print("--- OpenAI Client Initialized ---")
    except Exception as e:
        print(f"ERROR: Could not initialize OpenAI Client: {e}")
        openai_client = None

# --- Pydantic Schemas (API Contracts) ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model_id: Optional[str] = None

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    tokens_used: int
    sources: List[dict] = []

class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    sources: List[dict] = []

class ConversationResponse(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime

# --- Endpoints ---

@app.get("/health/")
def health_check():
    return {"status": "ok", "service": "sps-ai-fastapi"}

@app.post("/api/chat/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, session: Session = Depends(get_session)):
    user_msg_content = req.message
    conv_id = req.conversation_id

    # 1. Get or Create Conversation
    if conv_id:
        conversation = session.get(Conversation, conv_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Title logic: use first 50 chars of message
        title = user_msg_content[:50] + "..." if len(user_msg_content) > 50 else user_msg_content
        conversation = Conversation(title=title)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    # 2. Save User Message
    user_msg = Message(role="user", content=user_msg_content, conversation_id=conversation.id)
    session.add(user_msg)
    session.commit()

    # 3. Call OpenAI (Simple, no RAG for "minimal", but we can add system prompt)
    # Fetch history for context
    history = session.exec(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.created_at)).all()
    
    messages_payload = [{"role": "system", "content": "You are a helpful assistant for SPS Pipeline Simulator."}]
    for m in history:
        messages_payload.append({"role": m.role, "content": m.content})

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o", # or gpt-3.5-turbo check env later
            messages=messages_payload,
            temperature=0.7
        )
        ai_text = completion.choices[0].message.content
        usage = completion.usage.total_tokens if completion.usage else 0

    except Exception as e:
        ai_text = f"Error calling OpenAI: {str(e)}"
        usage = 0

    # 4. Save AI Message
    ai_msg = Message(
        role="assistant",
        content=ai_text,
        conversation_id=conversation.id,
        tokens_used=usage
    )
    session.add(ai_msg)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    session.add(conversation)
    session.commit()
    session.refresh(ai_msg)

    return ChatResponse(
        conversation_id=conversation.id,
        message_id=ai_msg.id,
        content=ai_text,
        sources=[]
    )

@app.get("/api/conversations/", response_model=List[ConversationResponse])
def list_conversations(session: Session = Depends(get_session)):
    conversations = session.exec(select(Conversation).order_by(Conversation.updated_at.desc())).all()
    response = []
    for c in conversations:
        # Count messages manually or via relation
        count = len(c.messages)
        response.append(ConversationResponse(
            id=c.id,
            title=c.title,
            message_count=count,
            created_at=c.created_at,
            updated_at=c.updated_at
        ))
    return response

@app.get("/api/conversations/{conversation_id}/", response_model=ConversationResponse)
def get_conversation(conversation_id: str, session: Session = Depends(get_session)):
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )

@app.get("/api/conversations/{conversation_id}/messages/", response_model=List[MessageResponse])
def get_messages(conversation_id: str, session: Session = Depends(get_session)):
    # Frontend expects {role, content, etc.}
    # Ensure checking coversation exists?
    messages = session.exec(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)).all()
    return [
        MessageResponse(
            id=m.id, 
            role=m.role, 
            content=m.content, 
            created_at=m.created_at, 
            tokens_used=m.tokens_used
        ) for m in messages
    ]

# Stub for Suggestions
@app.get("/api/suggestions/{model_id}/")
def get_suggestions(model_id: str):
    return {"suggestions": []}

@app.post("/api/suggestions/{model_id}/analyze/")
def analyze_suggestions(model_id: str):
    return {"suggestions": []}
