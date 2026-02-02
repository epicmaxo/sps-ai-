from sqlmodel import SQLModel, Field, Relationship, create_engine, Session
from typing import List, Optional
from datetime import datetime
import os
import uuid

# --- Database Setup ---
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./chat.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# --- Models ---
class Conversation(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: str = Field(default="New Conversation")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    messages: List["Message"] = Relationship(back_populates="conversation")

class Message(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    role: str
    content: str
    conversation_id: str = Field(foreign_key="conversation.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = Field(default=0)
    
    conversation: Conversation = Relationship(back_populates="messages")
