import asyncio
import time
import aiohttp
import json
from typing import List, Dict, Any, AsyncGenerator
from ..base import BaseAIProvider, ChatMessage, ChatResponse, MessageRole

class LocalProvider(BaseAIProvider):
    """Local LLM provider (Ollama) implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama2:7b')
        self.timeout = config.get('timeout', 30)
    
    async def generate_response(self, 
                              messages: List[ChatMessage],
                              max_tokens: int = 1000,
                              temperature: float = 0.7,
                              **kwargs) -> ChatResponse:
        """Generate response using local Ollama API"""
        start_time = time.time()
        
        # Convert messages to prompt format for local LLM
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Local LLM API error: {response.status}")
                    
                    result = await response.json()
                    response_time = int((time.time() - start_time) * 1000)
                    
                    return ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=result.get('response', '')
                        ),
                        finish_reason="stop",
                        usage={
                            "prompt_tokens": result.get('prompt_eval_count', 0),
                            "completion_tokens": result.get('eval_count', 0),
                            "total_tokens": result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                        },
                        model=self.model,
                        provider="local",
                        response_time_ms=response_time
                    )
                    
        except asyncio.TimeoutError:
            raise Exception("Local LLM timeout - is Ollama running?")
        except Exception as e:
            raise Exception(f"Local LLM error: {str(e)}")
    
    async def stream_response(self,
                            messages: List[ChatMessage], 
                            max_tokens: int = 1000,
                            temperature: float = 0.7,
                            **kwargs) -> AsyncGenerator[str, None]:
        """Stream response from local LLM"""
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    async for line in response.content:
                        if line.strip():
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    yield chunk['response']
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        embeddings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for text in texts:
                    payload = {
                        "model": self.model,
                        "prompt": text
                    }
                    
                    async with session.post(
                        f"{self.base_url}/api/embeddings",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            embeddings.append(result.get('embedding', []))
                        else:
                            # Fallback to dummy embedding if not supported
                            embeddings.append([0.0] * 768)  # Standard embedding size
                            
            return embeddings
            
        except Exception as e:
            # Return dummy embeddings if service unavailable
            return [[0.0] * 768 for _ in texts]
    
    def is_available(self) -> bool:
        """Check if local LLM is available"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def check():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        return response.status == 200
            
            return loop.run_until_complete(check())
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information"""
        return {
            "provider": "local",
            "model": self.model,
            "supports_functions": False,
            "supports_streaming": True,
            "context_window": 4096,  # Varies by model
            "cost_per_1k_tokens": {"input": 0.0, "output": 0.0}  # Free local inference
        }
    
    def supports_functions(self) -> bool:
        return False  # Most local models don't support function calling yet
        
    def supports_streaming(self) -> bool:
        return True
        
    def get_cost_per_token(self) -> Dict[str, float]:
        return {"input": 0.0, "output": 0.0}  # Local inference is free
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt for local LLMs"""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant: ")
        return "\n\n".join(prompt_parts)
