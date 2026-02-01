"""
=============================================================================
SPS Assistant - AI Models
=============================================================================
Database models for conversations, messages, and documents.
=============================================================================
"""

import uuid
from django.db import models
from django.conf import settings
from apps.core.models import TimeStampedModel


class Conversation(TimeStampedModel):
    """
    A chat conversation with the AI assistant.
    
    Conversations can be associated with a specific project/model
    to provide context-aware responses.
    """
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='ai_conversations',
    )
    
    # Optional context (Stored as IDs for stateless service)
    project_id = models.CharField(max_length=100, null=True, blank=True)
    model_id = models.CharField(max_length=100, null=True, blank=True)
    
    title = models.CharField(
        max_length=255,
        default='New Conversation',
    )
    
    # Metadata
    message_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Conversation'
        verbose_name_plural = 'Conversations'
    
    def __str__(self):
        return f"{self.title} - {self.user.email}"


class Message(TimeStampedModel):
    """
    A single message in a conversation.
    """
    
    class Role(models.TextChoices):
        USER = 'user', 'User'
        ASSISTANT = 'assistant', 'Assistant'
        SYSTEM = 'system', 'System'
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages',
    )
    
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
    )
    
    content = models.TextField()
    
    # Sources used for RAG responses
    sources = models.JSONField(
        default=list,
        blank=True,
        help_text="Document sources used to generate this response",
    )
    
    # Token usage tracking
    tokens_used = models.IntegerField(
        default=0,
        help_text="Number of tokens used for this message",
    )
    
    class Meta:
        ordering = ['created_at']
        verbose_name = 'Message'
        verbose_name_plural = 'Messages'
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class Document(TimeStampedModel):
    """
    A document indexed for RAG retrieval.
    
    Documents are chunked and embedded for semantic search.
    """
    
    class DocumentType(models.TextChoices):
        MANUAL = 'manual', 'SPS Manual'
        STANDARD = 'standard', 'Engineering Standard'
        GUIDELINE = 'guideline', 'Best Practice Guideline'
        TRAINING = 'training', 'Training Material'
        OTHER = 'other', 'Other'
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        INDEXING = 'indexing', 'Indexing'
        INDEXED = 'indexed', 'Indexed'
        FAILED = 'failed', 'Failed'
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='ai_documents',
    )
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    document_type = models.CharField(
        max_length=20,
        choices=DocumentType.choices,
        default=DocumentType.OTHER,
    )
    
    # File
    file = models.FileField(
        upload_to='ai_documents/',
    )
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    error_message = models.TextField(blank=True)
    
    # Indexing stats
    chunk_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Document'
        verbose_name_plural = 'Documents'
    
    def __str__(self):
        return self.name


class DocumentChunk(models.Model):
    """
    A chunk of a document with its embedding.
    
    Chunks are used for semantic search in RAG.
    """
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks',
    )
    
    content = models.TextField()
    
    # Position in document
    chunk_index = models.IntegerField()
    start_char = models.IntegerField(default=0)
    end_char = models.IntegerField(default=0)
    
    # Embedding stored as JSON array (for simplicity)
    # In production, use pgvector or dedicated vector store
    embedding = models.JSONField(
        default=list,
        blank=True,
    )
    
    class Meta:
        ordering = ['document', 'chunk_index']
        verbose_name = 'Document Chunk'
        verbose_name_plural = 'Document Chunks'
    
    def __str__(self):
        return f"{self.document.name} - Chunk {self.chunk_index}"


class Suggestion(TimeStampedModel):
    """
    A smart suggestion generated for a model.
    """
    
    class SuggestionType(models.TextChoices):
        WARNING = 'warning', 'Warning'
        ERROR = 'error', 'Error'
        OPTIMIZATION = 'optimization', 'Optimization'
        INFO = 'info', 'Information'
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    
    model_id = models.CharField(max_length=100)
    
    suggestion_type = models.CharField(
        max_length=20,
        choices=SuggestionType.choices,
    )
    
    element_type = models.CharField(
        max_length=50,
        blank=True,
        help_text="Type of element (node, pipe, pump, etc.)",
    )
    
    element_name = models.CharField(
        max_length=255,
        blank=True,
        help_text="Name of the affected element",
    )
    
    title = models.CharField(max_length=255)
    message = models.TextField()
    recommendation = models.TextField(blank=True)
    
    # Whether user has dismissed this suggestion
    is_dismissed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Suggestion'
        verbose_name_plural = 'Suggestions'
    
    def __str__(self):
        return f"{self.suggestion_type}: {self.title}"
