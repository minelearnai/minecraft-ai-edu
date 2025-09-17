import os
from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

class AIProvider(str, Enum):
    """Available AI providers"""
    OPENAI = "openai"
    LOCAL = "local" 
    ANTHROPIC = "anthropic"
    AUTO = "auto"  # Automatic provider selection

class AIConfig(BaseModel):
    """AI configuration settings"""
    
    # Provider selection
    default_provider: AIProvider = Field(default=AIProvider.AUTO)
    fallback_providers: list[AIProvider] = Field(default=[AIProvider.LOCAL, AIProvider.OPENAI])
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-3.5-turbo")
    openai_base_url: Optional[str] = Field(default=None)
    
    # Local LLM Configuration (Ollama)
    local_base_url: str = Field(default="http://localhost:11434")
    local_model: str = Field(default="llama2:7b")
    local_timeout: int = Field(default=30)
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None) 
    anthropic_model: str = Field(default="claude-3-sonnet-20240229")
    
    # Educational Content Settings
    educational_context: bool = Field(default=True)
    age_appropriate_filter: bool = Field(default=True)
    minecraft_focused: bool = Field(default=True)
    
    # Response Settings
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.7)
    response_timeout: int = Field(default=30)
    
    # Safety Settings
    content_filter_enabled: bool = Field(default=True)
    educational_guidelines: bool = Field(default=True)
    
    @classmethod
    def from_env(cls) -> 'AIConfig':
        """Create configuration from environment variables"""
        return cls(
            # Provider selection
            default_provider=AIProvider(os.getenv('AI_DEFAULT_PROVIDER', 'auto')),
            
            # OpenAI
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            openai_base_url=os.getenv('OPENAI_BASE_URL'),
            
            # Local LLM
            local_base_url=os.getenv('LOCAL_LLM_URL', 'http://localhost:11434'),
            local_model=os.getenv('LOCAL_MODEL', 'llama2:7b'),
            local_timeout=int(os.getenv('LOCAL_TIMEOUT', '30')),
            
            # Anthropic
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            anthropic_model=os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
            
            # Educational settings
            educational_context=os.getenv('EDUCATIONAL_CONTEXT', 'true').lower() == 'true',
            minecraft_focused=os.getenv('MINECRAFT_FOCUSED', 'true').lower() == 'true',
            
            # Response settings
            max_tokens=int(os.getenv('AI_MAX_TOKENS', '1000')),
            temperature=float(os.getenv('AI_TEMPERATURE', '0.7')),
            response_timeout=int(os.getenv('AI_TIMEOUT', '30')),
        )
        
    def get_provider_config(self, provider: AIProvider) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        if provider == AIProvider.OPENAI:
            return {
                'api_key': self.openai_api_key,
                'model': self.openai_model,
                'base_url': self.openai_base_url
            }
        elif provider == AIProvider.LOCAL:
            return {
                'base_url': self.local_base_url,
                'model': self.local_model,
                'timeout': self.local_timeout
            }
        elif provider == AIProvider.ANTHROPIC:
            return {
                'api_key': self.anthropic_api_key,
                'model': self.anthropic_model
            }
        else:
            return {}
            
    def is_provider_available(self, provider: AIProvider) -> bool:
        """Check if provider is properly configured"""
        if provider == AIProvider.OPENAI:
            return self.openai_api_key is not None
        elif provider == AIProvider.LOCAL:
            return True  # Local LLM can always attempt connection
        elif provider == AIProvider.ANTHROPIC:
            return self.anthropic_api_key is not None
        return False
