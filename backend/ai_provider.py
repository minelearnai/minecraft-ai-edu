from abc import ABC, abstractmethod
from typing import Optional
import os

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

class LocalLMProvider(IAIProvider):
    """Local Language Model Provider (Simplified)"""
    
    def __init__(self):
        print("🤖 Initializing Local LM Provider (Mock)")
    
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

def get_ai_provider() -> IAIProvider:
    """Factory function to get the configured AI provider"""
    use_local = os.getenv('USE_LOCAL_LM', '1') == '1'
    
    if use_local:
        return LocalLMProvider()
    else:
        # TODO: Implement OpenAIProvider when needed
        raise NotImplementedError("OpenAI provider not yet implemented")