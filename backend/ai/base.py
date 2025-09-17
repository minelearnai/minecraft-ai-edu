from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel
from enum import Enum

class MessageRole(str, Enum):
    """Message roles for chat conversations"""
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"
    FUNCTION = "function"

class ChatMessage(BaseModel):
    """Standardized chat message format"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Standardized response from AI providers"""
    message: ChatMessage
    finish_reason: str
    usage: Dict[str, int] = {}
    model: str
    provider: str
    response_time_ms: int

class BaseAIProvider(ABC):
    """Abstract base class for all AI providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__
    
    @abstractmethod
    async def generate_response(self, 
                              messages: List[ChatMessage],
                              max_tokens: int = 1000,
                              temperature: float = 0.7,
                              **kwargs) -> ChatResponse:
        """Generate a single response from the AI model"""
        pass
    
    @abstractmethod 
    async def stream_response(self,
                            messages: List[ChatMessage], 
                            max_tokens: int = 1000,
                            temperature: float = 0.7,
                            **kwargs) -> AsyncGenerator[str, None]:
        """Stream response tokens as they're generated"""
        pass
        
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text strings"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        pass
        
    def supports_functions(self) -> bool:
        """Whether this provider supports function calling"""
        return False
        
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses"""
        return False
        
    def get_cost_per_token(self) -> Dict[str, float]:
        """Get cost per input/output token (if applicable)"""
        return {"input": 0.0, "output": 0.0}
