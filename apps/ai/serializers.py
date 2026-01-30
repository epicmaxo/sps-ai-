"""
=============================================================================
SPS Assistant - AI Serializers
=============================================================================
API serializers for conversations, messages, and documents.
=============================================================================
"""

from rest_framework import serializers
from .models import Conversation, Message, Document, DocumentChunk, Suggestion


class MessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages."""
    
    class Meta:
        model = Message
        fields = ['id', 'role', 'content', 'sources', 'tokens_used', 'created_at']
        read_only_fields = ['id', 'sources', 'tokens_used', 'created_at']


class ConversationListSerializer(serializers.ModelSerializer):
    """Simplified serializer for conversation lists."""
    
    last_message = serializers.SerializerMethodField()
    model_name = serializers.CharField(source='model.name', read_only=True)
    project_name = serializers.CharField(source='project.name', read_only=True)
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'title', 'message_count', 'project_name', 'model_name',
            'last_message', 'created_at', 'updated_at',
        ]
    
    def get_last_message(self, obj):
        last = obj.messages.last()
        if last:
            return {
                'role': last.role,
                'content': last.content[:100] + ('...' if len(last.content) > 100 else ''),
                'created_at': last.created_at.isoformat(),
            }
        return None


class ConversationDetailSerializer(serializers.ModelSerializer):
    """Full serializer with messages."""
    
    messages = MessageSerializer(many=True, read_only=True)
    model_name = serializers.CharField(source='model.name', read_only=True)
    project_name = serializers.CharField(source='project.name', read_only=True)
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'title', 'message_count', 'project', 'model',
            'project_name', 'model_name', 'messages',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'message_count', 'created_at', 'updated_at']


class ConversationCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating conversations."""
    
    class Meta:
        model = Conversation
        fields = ['title', 'project', 'model']


class ChatMessageSerializer(serializers.Serializer):
    """Serializer for sending chat messages."""
    
    message = serializers.CharField(required=True)
    conversation_id = serializers.UUIDField(required=False)
    model_id = serializers.UUIDField(required=False)
    include_rag = serializers.BooleanField(default=True)


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for documents."""
    
    uploaded_by_email = serializers.CharField(source='uploaded_by.email', read_only=True)
    
    class Meta:
        model = Document
        fields = [
            'id', 'name', 'description', 'document_type', 'file',
            'status', 'error_message', 'chunk_count',
            'uploaded_by_email', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'status', 'error_message', 'chunk_count', 'created_at', 'updated_at']


class DocumentUploadSerializer(serializers.ModelSerializer):
    """Serializer for uploading documents."""
    
    class Meta:
        model = Document
        fields = ['name', 'description', 'document_type', 'file']


class SuggestionSerializer(serializers.ModelSerializer):
    """Serializer for suggestions."""
    
    class Meta:
        model = Suggestion
        fields = [
            'id', 'suggestion_type', 'element_type', 'element_name',
            'title', 'message', 'recommendation', 'is_dismissed',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']
