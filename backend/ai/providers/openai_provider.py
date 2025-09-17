import asyncio
import time
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI
from ..base import BaseAIProvider, ChatMessage, ChatResponse, MessageRole

class OpenAIProvider(BaseAIProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.get('api_key'),
            base_url=config.get('base_url')
        )
        self.model = config.get('model', 'gpt-3.5-turbo')
    
    async def generate_response(self, 
                              messages: List[ChatMessage],
                              max_tokens: int = 1000,
                              temperature: float = 0.7,
                              **kwargs) -> ChatResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()
        
        # Convert our messages to OpenAI format
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            response_time = int((time.time() - start_time) * 1000)
            
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.choices[0].message.content
                ),
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=response.model,
                provider="openai",
                response_time_ms=response_time
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def stream_response(self,
                            messages: List[ChatMessage], 
                            max_tokens: int = 1000,
                            temperature: float = 0.7,
                            **kwargs) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI"""
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise Exception(f"OpenAI embeddings error: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if OpenAI is configured and available"""
        return self.config.get('api_key') is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            "provider": "openai",
            "model": self.model,
            "supports_functions": True,
            "supports_streaming": True,
            "context_window": 4096 if "gpt-3.5" in self.model else 8192,
            "cost_per_1k_tokens": {
                "input": 0.0015 if "gpt-3.5" in self.model else 0.03,
                "output": 0.002 if "gpt-3.5" in self.model else 0.06
            }
        }
    
    def supports_functions(self) -> bool:
        return True
        
    def supports_streaming(self) -> bool:
        return True
        
    def get_cost_per_token(self) -> Dict[str, float]:
        """OpenAI pricing per token"""
        if "gpt-3.5" in self.model:
            return {"input": 0.0000015, "output": 0.000002}
        else:  # GPT-4
            return {"input": 0.00003, "output": 0.00006}
