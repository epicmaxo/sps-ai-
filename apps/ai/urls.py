"""
=============================================================================
SPS Assistant - AI URLs
=============================================================================
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ChatView, ConversationViewSet, DocumentViewSet,
    SuggestionView, AnalyzeModelView, DismissSuggestionView,
)

app_name = 'ai'

router = DefaultRouter()
router.register(r'conversations', ConversationViewSet, basename='conversation')
router.register(r'documents', DocumentViewSet, basename='document')

urlpatterns = [
    path('', include(router.urls)),
    
    # Chat
    path('chat/', ChatView.as_view(), name='chat'),
    
    # Suggestions
    path('suggestions/<uuid:model_id>/', SuggestionView.as_view(), name='suggestions'),
    path('suggestions/<uuid:model_id>/analyze/', AnalyzeModelView.as_view(), name='analyze'),
    path('suggestions/<uuid:suggestion_id>/dismiss/', DismissSuggestionView.as_view(), name='dismiss'),
]
