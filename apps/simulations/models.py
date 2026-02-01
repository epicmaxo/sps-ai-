"""
=============================================================================
SPS Assistant - Simulations Models
=============================================================================
Stub models for simulation jobs (to be implemented by sps_backend).
=============================================================================
"""

from django.db import models
from apps.core.models import BaseModel
from apps.models.models import PipelineModel


class SimulationJob(BaseModel):
    """
    A simulation job in the SPS system.
    
    This is a stub model - the full implementation exists in sps_backend.
    """
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
    
    model = models.ForeignKey(
        PipelineModel,
        on_delete=models.CASCADE,
        related_name='simulation_jobs',
    )
    
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Simulation results (simplified)
    result_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Simulation results and summary"
    )
    
    class Meta:
        app_label = 'simulations'
        ordering = ['-created_at']
        verbose_name = 'Simulation Job'
        verbose_name_plural = 'Simulation Jobs'
    
    def __str__(self):
        return f"{self.model.name} - {self.status}"
    
    @property
    def result(self):
        """Stub for result - returns object with summary attribute"""
        class StubResult:
            summary = {}
        return StubResult()
