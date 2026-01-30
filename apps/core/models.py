"""
=============================================================================
SPS Assistant - Core Models
=============================================================================
Base model classes and shared utilities.
=============================================================================
"""

from django.db import models
import uuid


class TimeStampedModel(models.Model):
    """
    Abstract base model that provides:
    - created_at: Timestamp when record was created
    - updated_at: Timestamp when record was last modified
    
    Usage:
        class MyModel(TimeStampedModel):
            name = models.CharField(max_length=100)
    """
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True


class UUIDModel(models.Model):
    """
    Abstract base model that uses UUID as primary key instead of integer.
    
    Benefits:
    - IDs are not sequential (harder to guess)
    - Safe for distributed systems
    - Better for API URLs
    
    Usage:
        class MyModel(UUIDModel):
            name = models.CharField(max_length=100)
    """
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    
    class Meta:
        abstract = True


class BaseModel(UUIDModel, TimeStampedModel):
    """
    Combines UUID primary key and timestamps.
    
    Most models in SPS Assistant should extend this class.
    
    Usage:
        class Project(BaseModel):
            name = models.CharField(max_length=100)
            # Automatically gets: id (UUID), created_at, updated_at
    """
    
    class Meta:
        abstract = True
