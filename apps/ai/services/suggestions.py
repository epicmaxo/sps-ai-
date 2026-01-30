"""
=============================================================================
SPS Assistant - Suggestions Service
=============================================================================
Generates smart suggestions for pipeline models.

LAYMAN EXPLANATION:
This service analyzes your pipeline model and simulation results
to find potential issues and optimization opportunities - like having
an experienced engineer review your work and point out things to check.
=============================================================================
"""

import logging
from typing import List, Dict, Any, Optional
from decimal import Decimal

from apps.models.models import PipelineModel
from apps.simulations.models import SimulationJob
from ..models import Suggestion
from .llm import get_llm_provider

logger = logging.getLogger(__name__)


# Engineering thresholds
VELOCITY_WARNING = 3.0      # m/s
VELOCITY_CRITICAL = 4.0     # m/s
PRESSURE_MARGIN = 0.9       # 90% of MAOP
PUMP_EFFICIENCY_MIN = 0.6   # 60%


class SuggestionService:
    """
    Generates smart suggestions for pipeline models.
    
    Two types of suggestions:
    1. Rule-based: Check against engineering limits
    2. AI-based: Use LLM for deeper analysis
    """
    
    def __init__(self):
        self.llm = get_llm_provider()
    
    def analyze_model(
        self,
        model: PipelineModel,
        include_ai: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Analyze a pipeline model and generate suggestions.
        
        Args:
            model: PipelineModel to analyze
            include_ai: Also run AI analysis
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Rule-based analysis
        suggestions.extend(self._check_velocities(model))
        suggestions.extend(self._check_pressures(model))
        suggestions.extend(self._check_pumps(model))
        suggestions.extend(self._check_model_structure(model))
        
        # AI-based analysis
        if include_ai:
            ai_suggestions = self._ai_analysis(model)
            suggestions.extend(ai_suggestions)
        
        # Save suggestions to database
        self._save_suggestions(model, suggestions)
        
        return suggestions
    
    def get_suggestions(
        self,
        model: PipelineModel,
        include_dismissed: bool = False
    ) -> List[Dict[str, Any]]:
        """Get existing suggestions for a model."""
        qs = Suggestion.objects.filter(model=model)
        
        if not include_dismissed:
            qs = qs.filter(is_dismissed=False)
        
        return [
            {
                'id': str(s.id),
                'type': s.suggestion_type,
                'element_type': s.element_type,
                'element_name': s.element_name,
                'title': s.title,
                'message': s.message,
                'recommendation': s.recommendation,
                'is_dismissed': s.is_dismissed,
            }
            for s in qs
        ]
    
    def dismiss_suggestion(self, suggestion_id: str):
        """Mark a suggestion as dismissed."""
        Suggestion.objects.filter(id=suggestion_id).update(is_dismissed=True)
    
    def _check_velocities(self, model: PipelineModel) -> List[Dict]:
        """Check pipe velocities against limits."""
        suggestions = []
        
        # Get results if available
        recent_job = SimulationJob.objects.filter(
            model=model,
            status='completed'
        ).order_by('-completed_at').first()
        
        if not recent_job or not hasattr(recent_job, 'result'):
            return suggestions
        
        pipe_results = recent_job.result.pipe_results or {}
        
        for pipe_name, data in pipe_results.items():
            velocity = data.get('velocity', 0)
            
            if velocity > VELOCITY_CRITICAL:
                suggestions.append({
                    'type': 'error',
                    'element_type': 'pipe',
                    'element_name': pipe_name,
                    'title': f'Critical velocity in {pipe_name}',
                    'message': f'Velocity of {velocity:.2f} m/s exceeds critical limit of {VELOCITY_CRITICAL} m/s. Risk of erosion-corrosion.',
                    'recommendation': 'Increase pipe diameter or reduce flow rate. Consider parallel piping.',
                })
            elif velocity > VELOCITY_WARNING:
                suggestions.append({
                    'type': 'warning',
                    'element_type': 'pipe',
                    'element_name': pipe_name,
                    'title': f'High velocity in {pipe_name}',
                    'message': f'Velocity of {velocity:.2f} m/s exceeds recommended {VELOCITY_WARNING} m/s limit.',
                    'recommendation': 'Consider increasing pipe diameter to reduce velocity.',
                })
        
        return suggestions
    
    def _check_pressures(self, model: PipelineModel) -> List[Dict]:
        """Check pressures against MAOP limits."""
        suggestions = []
        
        recent_job = SimulationJob.objects.filter(
            model=model,
            status='completed'
        ).order_by('-completed_at').first()
        
        if not recent_job or not hasattr(recent_job, 'result'):
            return suggestions
        
        node_results = recent_job.result.node_results or {}
        
        # Get node MAOP from model
        for node in model.nodes.all():
            if node.name not in node_results:
                continue
            
            pressure = node_results[node.name].get('pressure', 0)
            max_pressure = float(node.max_pressure or 0)
            
            if max_pressure > 0 and pressure > max_pressure:
                suggestions.append({
                    'type': 'error',
                    'element_type': 'node',
                    'element_name': node.name,
                    'title': f'Pressure exceeds MAOP at {node.name}',
                    'message': f'Operating pressure {pressure:.0f} kPa exceeds MAOP of {max_pressure:.0f} kPa.',
                    'recommendation': 'Review pressure control and relief systems. Consider adding pressure control valve.',
                })
            elif max_pressure > 0 and pressure > max_pressure * PRESSURE_MARGIN:
                suggestions.append({
                    'type': 'warning',
                    'element_type': 'node',
                    'element_name': node.name,
                    'title': f'Pressure approaching MAOP at {node.name}',
                    'message': f'Operating pressure {pressure:.0f} kPa is within 10% of MAOP ({max_pressure:.0f} kPa).',
                    'recommendation': 'Monitor pressure closely. Review relief capacity.',
                })
        
        return suggestions
    
    def _check_pumps(self, model: PipelineModel) -> List[Dict]:
        """Check pump operating conditions."""
        suggestions = []
        
        recent_job = SimulationJob.objects.filter(
            model=model,
            status='completed'
        ).order_by('-completed_at').first()
        
        if not recent_job or not hasattr(recent_job, 'result'):
            return suggestions
        
        pump_results = recent_job.result.pump_results or {}
        
        for pump_name, data in pump_results.items():
            efficiency = data.get('efficiency', 0)
            
            if 0 < efficiency < PUMP_EFFICIENCY_MIN:
                suggestions.append({
                    'type': 'optimization',
                    'element_type': 'pump',
                    'element_name': pump_name,
                    'title': f'Low efficiency at {pump_name}',
                    'message': f'Pump operating at {efficiency*100:.0f}% efficiency, below recommended minimum of {PUMP_EFFICIENCY_MIN*100:.0f}%.',
                    'recommendation': 'Operating point may be far from BEP. Consider VFD speed adjustment or impeller trim.',
                })
        
        return suggestions
    
    def _check_model_structure(self, model: PipelineModel) -> List[Dict]:
        """Check model structure for issues."""
        suggestions = []
        
        # Check for source
        if not model.nodes.filter(node_type='source').exists():
            suggestions.append({
                'type': 'error',
                'element_type': 'model',
                'element_name': model.name,
                'title': 'No source node defined',
                'message': 'Model has no source node. Simulation requires at least one source.',
                'recommendation': 'Add a source node to define system inlet.',
            })
        
        # Check for sink
        if not model.nodes.filter(node_type='sink').exists():
            suggestions.append({
                'type': 'error',
                'element_type': 'model',
                'element_name': model.name,
                'title': 'No sink node defined',
                'message': 'Model has no sink/delivery node. Simulation requires at least one sink.',
                'recommendation': 'Add a sink node to define system outlet.',
            })
        
        return suggestions
    
    def _ai_analysis(self, model: PipelineModel) -> List[Dict]:
        """Use AI to generate advanced suggestions."""
        suggestions = []
        
        try:
            # Build model summary
            summary = self._build_model_summary(model)
            
            prompt = f"""Analyze this pipeline model and provide optimization suggestions:

{summary}

Provide 2-3 specific, actionable suggestions for improving this pipeline system.
For each suggestion, provide:
1. A brief title
2. The issue or opportunity
3. A specific recommendation

Focus on hydraulic efficiency, safety, and operational reliability.
Format as JSON array: [{{"title": "...", "message": "...", "recommendation": "..."}}]
"""
            
            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a pipeline hydraulics expert. Provide practical engineering suggestions.",
                max_tokens=1024,
                temperature=0.3,
            )
            
            # Parse response
            import json
            content = response['content']
            
            # Extract JSON from response
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                ai_suggestions = json.loads(content[start:end])
                
                for s in ai_suggestions:
                    suggestions.append({
                        'type': 'optimization',
                        'element_type': 'model',
                        'element_name': model.name,
                        'title': s.get('title', 'AI Suggestion'),
                        'message': s.get('message', ''),
                        'recommendation': s.get('recommendation', ''),
                    })
                    
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
        
        return suggestions
    
    def _build_model_summary(self, model: PipelineModel) -> str:
        """Build a text summary of the model."""
        parts = [
            f"Model: {model.name}",
            f"Fluid: {model.fluid_type}",
            f"Nodes: {model.nodes.count()}",
            f"Pipes: {model.pipes.count()}",
            f"Pumps: {model.pumps.count()}",
            f"Valves: {model.valves.count()}",
        ]
        
        # Add simulation results if available
        recent_job = SimulationJob.objects.filter(
            model=model,
            status='completed'
        ).order_by('-completed_at').first()
        
        if recent_job and hasattr(recent_job, 'result'):
            summary = recent_job.result.summary or {}
            if summary:
                parts.append(f"\nSimulation Results:")
                for key, value in summary.items():
                    parts.append(f"  {key}: {value}")
        
        return '\n'.join(parts)
    
    def _save_suggestions(self, model: PipelineModel, suggestions: List[Dict]):
        """Save suggestions to database."""
        # Clear old suggestions
        Suggestion.objects.filter(model=model).delete()
        
        # Create new suggestions
        for s in suggestions:
            Suggestion.objects.create(
                model=model,
                suggestion_type=s['type'],
                element_type=s.get('element_type', ''),
                element_name=s.get('element_name', ''),
                title=s['title'],
                message=s['message'],
                recommendation=s.get('recommendation', ''),
            )
