"""
=============================================================================
SPS Assistant - LLM Service
=============================================================================
Abstraction layer for LLM providers (Claude, OpenAI, etc.)
=============================================================================
"""

import os
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the LLM."""
        pass


class AnthropicProvider(LLMProvider):
    """Claude/Anthropic provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response using Claude."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
            )
            
            return {
                'content': response.content[0].text,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'model': self.model,
                'stop_reason': response.stop_reason,
            }
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response using OpenAI."""
        try:
            # Prepend system message
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=full_messages,
            )
            
            return {
                'content': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens,
                'model': self.model,
                'stop_reason': response.choices[0].finish_reason,
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a mock response."""
        last_message = messages[-1]['content'] if messages else ''
        
        return {
            'content': f"This is a mock response to: {last_message[:100]}...",
            'tokens_used': 100,
            'model': 'mock',
            'stop_reason': 'end_turn',
        }


def get_llm_provider(
    provider: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.
    
    Args:
        provider: Provider name ('anthropic', 'openai', 'mock')
        **kwargs: Additional arguments passed to provider
        
    Returns:
        LLMProvider instance
    """
    provider = provider or os.environ.get('AI_PROVIDER', 'anthropic')
    
    providers = {
        'anthropic': AnthropicProvider,
        'openai': OpenAIProvider,
        'mock': MockProvider,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
    
    return providers[provider](**kwargs)
