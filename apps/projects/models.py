"""
=============================================================================
SPS Assistant - Projects Models
=============================================================================
Stub models for project management (to be implemented by sps_backend).
=============================================================================
"""

from django.db import models
from django.conf import settings
from apps.core.models import BaseModel


class Project(BaseModel):
    """
    A project in the SPS system.
    
    This is a stub model - the full implementation exists in sps_backend.
    """
    
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='projects',
    )
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Project'
        verbose_name_plural = 'Projects'
    
    def __str__(self):
        return self.name
