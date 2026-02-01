"""
=============================================================================
SPS Assistant - Chat Service
=============================================================================
Handles AI chat interactions with context awareness.
=============================================================================
"""

import logging
from typing import Dict, Any, List, Optional


from django.conf import settings

from ..models import Conversation, Message
from .llm import get_llm_provider
from .rag import RAGService


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert pipeline hydraulics engineer assistant for SPS (Synergi Pipeline Simulator). You help users:

- Understand their pipeline models
- Interpret simulation results
- Troubleshoot hydraulic issues
- Apply engineering standards (SAES, ASME, API)
- Optimize system performance

Guidelines:
1. Be specific and reference actual model elements when possible
2. Cite engineering standards when applicable
3. Provide actionable recommendations
4. Explain the engineering reasoning
5. Use appropriate units (SI preferred)
6. Warn about potential safety issues

{context}
"""


class ChatService:
    """
    Handles AI chat interactions.
    
    The service:
    1. Builds context from the current model
    2. Retrieves relevant documents via RAG
    3. Generates responses via LLM
    4. Saves conversation history
    """
    
    def __init__(self):
        self.llm = get_llm_provider()
        self.rag = RAGService()
    
    def process_message(
        self,
        conversation: Conversation,
        user_message: str,
        include_model_context: bool = True,
        include_rag: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a user message and generate AI response.
        
        Args:
            conversation: Conversation object
            user_message: The user's message
            include_model_context: Include pipeline model data
            include_rag: Include RAG document retrieval
            
        Returns:
            Dict with response content and metadata
        """
        # Save user message
        user_msg = Message.objects.create(
            conversation=conversation,
            role=Message.Role.USER,
            content=user_message,
        )
        
        # Build context
        context_parts = []
        
        # Model context
        if include_model_context and conversation.model:
            model_context = self._build_model_context(conversation.model)
            context_parts.append(model_context)
        
        # RAG context
        sources = []
        if include_rag:
            rag_results = self.rag.retrieve(user_message)
            if rag_results:
                rag_context = self._build_rag_context(rag_results)
                context_parts.append(rag_context)
                sources = [
                    {'document': r['document_name'], 'chunk': r['chunk_index']}
                    for r in rag_results
                ]
        

        # Build system prompt
        system_prompt = SYSTEM_PROMPT.format(context="")
        
        # Get conversation history
        messages = self._get_conversation_messages(conversation)
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        try:
            response = self.llm.generate(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.7,
            )
            
            assistant_content = response['content']
            tokens_used = response.get('tokens_used', 0)
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            assistant_content = "I apologize, but I encountered an error processing your request. Please try again."
            tokens_used = 0
        
        # Save assistant message
        assistant_msg = Message.objects.create(
            conversation=conversation,
            role=Message.Role.ASSISTANT,
            content=assistant_content,
            sources=sources,
            tokens_used=tokens_used,
        )
        
        # Update conversation
        conversation.message_count = conversation.messages.count()
        if conversation.title == 'New Conversation':
            conversation.title = user_message[:50] + ('...' if len(user_message) > 50 else '')
        conversation.save()
        
        return {
            'message_id': str(assistant_msg.id),
            'content': assistant_content,
            'sources': sources,
            'tokens_used': tokens_used,
        }
    
    def _get_conversation_messages(
        self,
        conversation: Conversation,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent messages for conversation history."""
        messages = conversation.messages.filter(
            role__in=[Message.Role.USER, Message.Role.ASSISTANT]
        ).order_by('-created_at')[:max_messages]
        
        # Reverse to chronological order
        messages = list(messages)[::-1]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
