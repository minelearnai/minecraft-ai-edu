import asyncio
from typing import Optional, List
from .base import BaseAIProvider
from .config import AIConfig, AIProvider
from .providers import OpenAIProvider, LocalProvider, AnthropicProvider

class AIProviderFactory:
    """Factory for creating and managing AI providers"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self._providers = {}
        self._initialized_providers = {}
    
    async def get_provider(self, provider_name: Optional[AIProvider] = None) -> BaseAIProvider:
        """Get AI provider instance with automatic fallback"""
        if provider_name is None:
            provider_name = self.config.default_provider
        
        # Handle AUTO provider selection
        if provider_name == AIProvider.AUTO:
            provider_name = await self._select_best_provider()
        
        # Return cached provider if available
        if provider_name in self._initialized_providers:
            return self._initialized_providers[provider_name]
        
        # Create new provider
        provider = await self._create_provider(provider_name)
        if provider:
            self._initialized_providers[provider_name] = provider
            return provider
        
        # Fallback to next available provider
        for fallback in self.config.fallback_providers:
            if fallback != provider_name:
                try:
                    provider = await self._create_provider(fallback)
                    if provider:
                        self._initialized_providers[fallback] = provider
                        print(f"⚠️  Falling back to {fallback.value} provider")
                        return provider
                except Exception as e:
                    print(f"❌ Fallback {fallback.value} failed: {e}")
                    continue
        
        raise Exception("No AI providers are available")
    
    async def _create_provider(self, provider_name: AIProvider) -> Optional[BaseAIProvider]:
        """Create a specific AI provider"""
        provider_config = self.config.get_provider_config(provider_name)
        
        try:
            if provider_name == AIProvider.OPENAI:
                provider = OpenAIProvider(provider_config)
            elif provider_name == AIProvider.LOCAL:
                provider = LocalProvider(provider_config)
            elif provider_name == AIProvider.ANTHROPIC:
                provider = AnthropicProvider(provider_config)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            # Test provider availability
            if not provider.is_available():
                print(f"⚠️  Provider {provider_name.value} is not properly configured")
                return None
            
            print(f"✅ Initialized {provider_name.value} provider")
            return provider
            
        except Exception as e:
            print(f"❌ Failed to initialize {provider_name.value}: {e}")
            return None
    
    async def _select_best_provider(self) -> AIProvider:
        """Automatically select the best available provider"""
        # Priority order for automatic selection
        priority_order = [AIProvider.OPENAI, AIProvider.LOCAL, AIProvider.ANTHROPIC]
        
        for provider in priority_order:
            if self.config.is_provider_available(provider):
                # Quick availability check
                test_provider = await self._create_provider(provider)
                if test_provider and test_provider.is_available():
                    print(f"🤖 Auto-selected {provider.value} provider")
                    return provider
        
        # Default to local if nothing else works
        print("🤖 Auto-selected local provider as fallback")
        return AIProvider.LOCAL
    
    async def get_all_available_providers(self) -> List[AIProvider]:
        """Get list of all available providers"""
        available = []
        
        for provider in AIProvider:
            if provider == AIProvider.AUTO:
                continue
                
            if self.config.is_provider_available(provider):
                test_provider = await self._create_provider(provider)
                if test_provider and test_provider.is_available():
                    available.append(provider)
        
        return available
    
    def get_provider_status(self) -> dict:
        """Get status of all configured providers"""
        status = {}
        
        for provider in AIProvider:
            if provider == AIProvider.AUTO:
                continue
                
            status[provider.value] = {
                "configured": self.config.is_provider_available(provider),
                "initialized": provider in self._initialized_providers,
                "model_info": None
            }
            
            if provider in self._initialized_providers:
                try:
                    status[provider.value]["model_info"] = self._initialized_providers[provider].get_model_info()
                except:
                    pass
        
        return status
    
    async def test_all_providers(self) -> dict:
        """Test all providers with a simple query"""
        results = {}
        test_messages = [
            {
                "role": "user", 
                "content": "Hello! Can you respond with just 'OK' to test the connection?"
            }
        ]
        
        for provider in AIProvider:
            if provider == AIProvider.AUTO:
                continue
                
            try:
                provider_instance = await self._create_provider(provider)
                if provider_instance:
                    # Convert dict messages to ChatMessage objects
                    from .base import ChatMessage, MessageRole
                    chat_messages = [ChatMessage(role=MessageRole.USER, content=test_messages[0]["content"])]
                    
                    response = await provider_instance.generate_response(
                        messages=chat_messages,
                        max_tokens=10,
                        temperature=0.1
                    )
                    
                    results[provider.value] = {
                        "status": "success",
                        "response": response.message.content,
                        "response_time_ms": response.response_time_ms,
                        "model": response.model
                    }
                else:
                    results[provider.value] = {
                        "status": "failed",
                        "error": "Provider not available or misconfigured"
                    }
                    
            except Exception as e:
                results[provider.value] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
