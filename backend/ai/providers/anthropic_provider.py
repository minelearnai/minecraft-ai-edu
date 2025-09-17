import asyncio
import time
from typing import List, Dict, Any, AsyncGenerator, Optional
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None
from ..base import BaseAIProvider, ChatMessage, ChatResponse, MessageRole

class AnthropicProvider(BaseAIProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if AsyncAnthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
            
        self.client = AsyncAnthropic(
            api_key=config.get('api_key')
        )
        self.model = config.get('model', 'claude-3-sonnet-20240229')
    
    async def generate_response(self, 
                              messages: List[ChatMessage],
                              max_tokens: int = 1000,
                              temperature: float = 0.7,
                              **kwargs) -> ChatResponse:
        """Generate response using Anthropic Claude API"""
        start_time = time.time()
        
        # Convert our messages to Anthropic format
        # Anthropic requires system message separately
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": "user" if msg.role == MessageRole.USER else "assistant",
                    "content": msg.content
                })
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message if system_message else "You are a helpful AI assistant.",
                messages=anthropic_messages,
                **kwargs
            )
            
            response_time = int((time.time() - start_time) * 1000)
            
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.content[0].text if response.content else ""
                ),
                finish_reason=response.stop_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=response.model,
                provider="anthropic",
                response_time_ms=response_time
            )
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def stream_response(self,
                            messages: List[ChatMessage], 
                            max_tokens: int = 1000,
                            temperature: float = 0.7,
                            **kwargs) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic"""
        # Convert messages
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": "user" if msg.role == MessageRole.USER else "assistant",
                    "content": msg.content
                })
        
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message if system_message else "You are a helpful AI assistant.",
                messages=anthropic_messages,
                **kwargs
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        yield event.delta.text
                        
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Anthropic doesn't provide embeddings API, return dummy embeddings"""
        # Return dummy embeddings since Anthropic doesn't have embeddings API
        return [[0.0] * 768 for _ in texts]  # Standard embedding dimension
    
    def is_available(self) -> bool:
        """Check if Anthropic is configured and available"""
        return self.config.get('api_key') is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information"""
        return {
            "provider": "anthropic",
            "model": self.model,
            "supports_functions": False,  # Claude doesn't support function calling yet
            "supports_streaming": True,
            "context_window": 200000,  # Claude 3 has large context window
            "cost_per_1k_tokens": {
                "input": 0.015,   # Claude 3 Sonnet pricing
                "output": 0.075
            }
        }
    
    def supports_functions(self) -> bool:
        return False  # Claude doesn't support function calling yet
        
    def supports_streaming(self) -> bool:
        return True
        
    def get_cost_per_token(self) -> Dict[str, float]:
        """Anthropic pricing per token (Claude 3 Sonnet)"""
        return {"input": 0.000015, "output": 0.000075}
