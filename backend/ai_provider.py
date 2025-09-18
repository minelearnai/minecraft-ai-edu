"""AI Provider abstraction layer supporting multiple AI services."""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class AIResponse:
    """Represents an AI response."""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class IAIProvider(ABC):
    """Abstract interface for AI providers"""
    
    @abstractmethod
    async def generate_response(self, user_id: str, prompt: str) -> str:
        """Generate a response to the given prompt"""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embeddings for the given text"""
        pass

class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.replace('Provider', '').lower()
    
    @abstractmethod
    async def generate_response(self, messages: List[ChatMessage], **kwargs) -> AIResponse:
        """Generate a response from the AI provider."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass
    
    def get_model_name(self) -> str:
        """Get the model name for this provider."""
        return self.config.get('model', 'unknown')

class LocalLMProvider(IAIProvider):
    """Local Language Model Provider (Simplified)"""
    
    def __init__(self):
        logger.info("🤖 Initializing Local LM Provider (Mock)")
    
    async def generate_response(self, user_id: str, prompt: str) -> str:
        # Simplified mock response for MVP
        # In production, this would use HuggingFace Transformers or similar
        if "matematyka" in prompt.lower() or "mathematics" in prompt.lower():
            return "To świetne pytanie z matematyki! W Minecraft możemy wizualizować geometrię budując różne kształty. Czy chciałbyś/chciałabyś zbudować konkretną figurę geometryczną?"
        elif "fizyka" in prompt.lower() or "physics" in prompt.lower():
            return "Fizyka w Minecraft to fascynujący temat! Możemy eksplorować grawitację, ruch, czy nawet elektryczność za pomocą redstone. O jakim zjawisku fizycznym chciałbyś/chciałabyś się dowiedzieć więcej?"
        elif "programowanie" in prompt.lower() or "programming" in prompt.lower():
            return "Programowanie w Minecraft używa bloków komend i logiki! Możemy tworzyć automatyczne systemy i maszyny. Czy chcesz nauczyć się o instrukcjach warunkowych (if) czy pętlach?"
        else:
            return f"Dzięki za pytanie! Jestem tutaj, żeby pomóc Ci w nauce. Możesz zapytać mnie o matematykę, fizykę, lub programowanie w kontekście Minecraft. Jak mogę Ci dzisiaj pomóc?"
    
    async def embed_text(self, text: str) -> list[float]:
        # Simplified mock embeddings
        # In production, this would use sentence-transformers or similar
        import hashlib
        import struct
        
        # Create deterministic "embeddings" based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to floats (simplified)
        embeddings = []
        for i in range(0, len(hash_bytes), 4):
            if i + 4 <= len(hash_bytes):
                val = struct.unpack('f', hash_bytes[i:i+4])[0]
                embeddings.append(val)
        
        # Pad or truncate to 384 dimensions
        while len(embeddings) < 384:
            embeddings.append(0.0)
        
        return embeddings[:384]

class OpenAIProvider(AIProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.model = config.get('model', 'gpt-3.5-turbo')
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    async def generate_response(self, messages: List[ChatMessage], **kwargs) -> AIResponse:
        try:
            # Convert ChatMessage objects to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} 
                for msg in messages
            ]
            
            # Add educational context if not present
            if not any(msg.role == 'system' for msg in messages):
                system_msg = {
                    "role": "system", 
                    "content": "You are an AI tutor for Minecraft Education. Help students learn programming, problem-solving, and STEM concepts through Minecraft. Be encouraging, clear, and educational."
                }
                openai_messages.insert(0, system_msg)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                provider=self.name,
                model=self.model,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                } if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")
    
    async def is_available(self) -> bool:
        try:
            # Quick test request to verify API key and connectivity
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning(f"OpenAI provider not available: {e}")
            return False

class LocalProvider(AIProvider):
    """Local LLM provider using Ollama."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama2:7b')
        
        # Try to import ollama, but don't fail if not available
        try:
            import ollama
            self.ollama = ollama
            self.client = ollama.AsyncClient(host=self.base_url)
        except ImportError:
            logger.warning("Ollama not installed. Local provider will not be available.")
            self.ollama = None
            self.client = None
    
    async def generate_response(self, messages: List[ChatMessage], **kwargs) -> AIResponse:
        if not self.client:
            raise RuntimeError("Ollama client not available. Install ollama-python package.")
        
        try:
            # Convert messages to Ollama format
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Add system message if not present
            if not any(msg.role == 'system' for msg in messages):
                system_msg = {
                    "role": "system", 
                    "content": "You are an AI tutor for Minecraft Education. Help students learn programming, problem-solving, and STEM concepts through Minecraft. Be encouraging, clear, and educational."
                }
                ollama_messages.insert(0, system_msg)
            
            response = await self.client.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    'temperature': kwargs.get('temperature', 0.7),
                    'num_predict': kwargs.get('max_tokens', 1000)
                }
            )
            
            return AIResponse(
                content=response['message']['content'],
                provider=self.name,
                model=self.model
            )
            
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise RuntimeError(f"Local LLM generation failed: {str(e)}")
    
    async def is_available(self) -> bool:
        if not self.client:
            return False
        
        try:
            # Check if Ollama is running and model is available
            models = await self.client.list()
            available_models = [model['name'] for model in models['models']]
            return self.model in available_models
        except Exception as e:
            logger.warning(f"Local provider not available: {e}")
            return False

class AIProviderManager:
    """Manages multiple AI providers with fallback support."""
    
    def __init__(self):
        self.providers: Dict[str, AIProvider] = {}
        self.default_provider = None
        self.fallback_order = []
        
        # Load configuration from environment
        self._load_configuration()
    
    def _load_configuration(self):
        """Load AI provider configuration from environment variables."""
        configs = {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            },
            'local': {
                'base_url': os.getenv('LOCAL_LLM_URL', 'http://localhost:11434'),
                'model': os.getenv('LOCAL_MODEL', 'llama2:7b')
            }
        }
        
        # Initialize providers
        provider_classes = {
            'openai': OpenAIProvider,
            'local': LocalProvider
        }
        
        for name, provider_class in provider_classes.items():
            config = configs.get(name, {})
            try:
                if name == 'openai' and config.get('api_key'):
                    self.providers[name] = provider_class(config)
                elif name == 'local':
                    # Always try to initialize local provider
                    self.providers[name] = provider_class(config)
            except Exception as e:
                logger.warning(f"Failed to initialize {name} provider: {e}")
        
        # Set default provider and fallback order
        self.default_provider = os.getenv('AI_DEFAULT_PROVIDER', 'local')
        self.fallback_order = ['openai', 'local']
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available and configured providers."""
        available = []
        for name, provider in self.providers.items():
            try:
                if await provider.is_available():
                    available.append(name)
            except Exception as e:
                logger.warning(f"Provider {name} availability check failed: {e}")
        return available
    
    async def generate_response(
        self, 
        messages: List[ChatMessage], 
        provider: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """Generate response using specified provider or with fallback."""
        
        # Determine which providers to try
        if provider and provider in self.providers:
            providers_to_try = [provider]
        else:
            # Use fallback order, starting with default
            providers_to_try = [self.default_provider] if self.default_provider else []
            providers_to_try.extend([p for p in self.fallback_order if p != self.default_provider])
        
        last_error = None
        
        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue
                
            provider_instance = self.providers[provider_name]
            
            try:
                if not await provider_instance.is_available():
                    logger.warning(f"Provider {provider_name} is not available")
                    continue
                    
                logger.info(f"Using AI provider: {provider_name}")
                return await provider_instance.generate_response(messages, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.error(f"Provider {provider_name} failed: {e}")
                continue
        
        # If we get here, all providers failed
        error_msg = f"All AI providers failed. Last error: {last_error}" if last_error else "No AI providers available"
        raise RuntimeError(error_msg)
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                'configured': True,
                'model': provider.get_model_name(),
                'initialized': provider is not None
            }
        return status

# Global provider manager instance
ai_manager = AIProviderManager()

def get_ai_provider() -> IAIProvider:
    """Factory function to get the configured AI provider"""
    use_local = os.getenv('USE_LOCAL_LM', '1') == '1'
    
    if use_local:
        return LocalLMProvider()
    else:
        # For compatibility with existing code, return a wrapper
        class OpenAIWrapper(IAIProvider):
            def __init__(self):
                self.manager = ai_manager
            
            async def generate_response(self, user_id: str, prompt: str) -> str:
                messages = [ChatMessage(role='user', content=prompt)]
                response = await self.manager.generate_response(messages, provider='openai')
                return response.content
            
            async def embed_text(self, text: str) -> list[float]:
                # Mock implementation - would need separate embedding service
                return [0.0] * 384
        
        try:
            return OpenAIWrapper()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI wrapper: {e}")
            logger.info("Falling back to local provider")
            return LocalLMProvider()

# Convenience functions for new interface
async def generate_ai_response(
    message: str,
    provider: Optional[str] = None,
    context: Optional[str] = None,
    **kwargs
) -> AIResponse:
    """Generate AI response from a simple message."""
    messages = []
    
    if context:
        messages.append(ChatMessage(role='system', content=context))
    
    messages.append(ChatMessage(role='user', content=message))
    
    return await ai_manager.generate_response(messages, provider=provider, **kwargs)

async def get_provider_info() -> Dict[str, Any]:
    """Get information about available providers."""
    available = await ai_manager.get_available_providers()
    status = ai_manager.get_provider_status()
    
    return {
        'default_provider': ai_manager.default_provider,
        'available_providers': available,
        'provider_status': status
    }

if __name__ == "__main__":
    # Test the AI providers
    async def test_providers():
        print("Testing AI providers...")
        
        info = await get_provider_info()
        print(f"Available providers: {info['available_providers']}")
        print(f"Default provider: {info['default_provider']}")
        
        if info['available_providers']:
            response = await generate_ai_response(
                "Hello! Can you help me with Minecraft Education?",
                context="You are a helpful AI tutor for Minecraft Education."
            )
            print(f"\nResponse from {response.provider}:")
            print(response.content)
        else:
            print("No providers available for testing.")
    
    asyncio.run(test_providers())