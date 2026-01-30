"""
=============================================================================
SPS Assistant - AI Views
=============================================================================
API endpoints for chat, documents, and suggestions.
=============================================================================
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, JSONParser
from django.shortcuts import get_object_or_404

from apps.models.models import PipelineModel
from .models import Conversation, Message, Document, Suggestion
from .serializers import (
    ConversationListSerializer, ConversationDetailSerializer,
    ConversationCreateSerializer, MessageSerializer, ChatMessageSerializer,
    DocumentSerializer, DocumentUploadSerializer,
    SuggestionSerializer,
)
from .services import ChatService, RAGService, SuggestionService


class ChatView(APIView):
    """Send a message and get AI response."""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = ChatMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        
        conversation_id = data.get('conversation_id')
        model_id = data.get('model_id')
        
        if conversation_id:
            conversation = get_object_or_404(
                Conversation, id=conversation_id, user=request.user,
            )
        else:
            model = None
            project = None
            if model_id:
                model = get_object_or_404(PipelineModel, id=model_id)
                project = model.project
            
            conversation = Conversation.objects.create(
                user=request.user, project=project, model=model,
            )
        
        chat_service = ChatService()
        result = chat_service.process_message(
            conversation=conversation,
            user_message=data['message'],
            include_model_context=bool(conversation.model),
            include_rag=data.get('include_rag', True),
        )
        
        return Response({
            'conversation_id': str(conversation.id),
            'message_id': result['message_id'],
            'content': result['content'],
            'sources': result.get('sources', []),
        })


class ConversationViewSet(viewsets.ModelViewSet):
    """ViewSet for managing conversations."""
    
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).select_related('project', 'model')
    
    def get_serializer_class(self):
        if self.action == 'list':
            return ConversationListSerializer
        if self.action == 'create':
            return ConversationCreateSerializer
        return ConversationDetailSerializer
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        conversation = self.get_object()
        messages = conversation.messages.all()
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)


class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for managing documents."""
    
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, JSONParser]
    
    def get_queryset(self):
        return Document.objects.filter(uploaded_by=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return DocumentUploadSerializer
        return DocumentSerializer
    
    def perform_create(self, serializer):
        serializer.save(uploaded_by=self.request.user)
    
    @action(detail=True, methods=['post'])
    def index(self, request, pk=None):
        document = self.get_object()
        
        if document.status == Document.Status.INDEXING:
            return Response({'error': 'Already indexing'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            rag_service = RAGService()
            chunk_count = rag_service.index_document(document)
            return Response({'status': 'indexed', 'chunk_count': chunk_count})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SuggestionView(APIView):
    """Get suggestions for a model."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, model_id):
        model = get_object_or_404(PipelineModel, id=model_id, project__owner=request.user)
        service = SuggestionService()
        suggestions = service.get_suggestions(model)
        return Response({
            'model_id': str(model.id),
            'suggestions': suggestions,
            'count': len(suggestions),
        })


class AnalyzeModelView(APIView):
    """Run analysis and generate suggestions."""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request, model_id):
        model = get_object_or_404(PipelineModel, id=model_id, project__owner=request.user)
        include_ai = request.data.get('include_ai', False)
        
        service = SuggestionService()
        suggestions = service.analyze_model(model, include_ai=include_ai)
        
        return Response({
            'model_id': str(model.id),
            'suggestions': suggestions,
            'count': len(suggestions),
        })


class DismissSuggestionView(APIView):
    """Dismiss a suggestion."""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request, suggestion_id):
        suggestion = get_object_or_404(
            Suggestion, id=suggestion_id, model__project__owner=request.user,
        )
        suggestion.is_dismissed = True
        suggestion.save()
        return Response({'status': 'dismissed'})
