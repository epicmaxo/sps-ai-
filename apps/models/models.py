"""
=============================================================================
SPS Assistant - Pipeline Models
=============================================================================
Stub models for pipeline modeling (to be implemented by sps_backend).
=============================================================================
"""

from django.db import models
from apps.core.models import BaseModel
from apps.projects.models import Project


class PipelineModel(BaseModel):
    """
    A pipeline model in the SPS system.
    
    This is a stub model - the full implementation exists in sps_backend.
    """
    
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='pipeline_models',
    )
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Fluid properties
    fluid_type = models.CharField(
        max_length=50,
        default='water',
        help_text="Type of fluid (water, oil, gas, etc.)"
    )
    
    # Model data (simplified)
    model_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Pipeline model configuration and data"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Pipeline Model'
        verbose_name_plural = 'Pipeline Models'
    
    def __str__(self):
        return f"{self.project.name} - {self.name}"
    
    # Stub properties to prevent AttributeErrors
    @property
    def nodes(self):
        """Stub for nodes - returns empty queryset"""
        return self.__class__.objects.none()
    
    @property
    def pipes(self):
        """Stub for pipes - returns empty queryset"""
        return self.__class__.objects.none()
    
    @property
    def pumps(self):
        """Stub for pumps - returns empty queryset"""
        return self.__class__.objects.none()
    
    @property
    def valves(self):
        """Stub for valves - returns empty queryset"""
        return self.__class__.objects.none()
    
    @property
    def tanks(self):
        """Stub for tanks - returns empty queryset"""
        return self.__class__.objects.none()
