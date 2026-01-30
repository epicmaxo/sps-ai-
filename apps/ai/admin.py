"""
=============================================================================
SPS Assistant - AI Admin
=============================================================================
"""

from django.contrib import admin
from .models import Conversation, Message, Document, DocumentChunk, Suggestion


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ['role', 'content', 'created_at']
    can_delete = False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'project', 'model', 'message_count', 'created_at']
    list_filter = ['created_at', 'user']
    search_fields = ['title', 'user__email']
    readonly_fields = ['id', 'message_count', 'created_at', 'updated_at']
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'role', 'content_preview', 'tokens_used', 'created_at']
    list_filter = ['role', 'created_at']
    readonly_fields = ['id', 'created_at']
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ['chunk_index', 'content']
    can_delete = False


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['name', 'document_type', 'status', 'chunk_count', 'uploaded_by', 'created_at']
    list_filter = ['status', 'document_type', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['id', 'status', 'chunk_count', 'created_at', 'updated_at']
    inlines = [DocumentChunkInline]


@admin.register(Suggestion)
class SuggestionAdmin(admin.ModelAdmin):
    list_display = ['title', 'suggestion_type', 'model', 'element_name', 'is_dismissed', 'created_at']
    list_filter = ['suggestion_type', 'is_dismissed', 'created_at']
    search_fields = ['title', 'message', 'element_name']
    readonly_fields = ['id', 'created_at', 'updated_at']
