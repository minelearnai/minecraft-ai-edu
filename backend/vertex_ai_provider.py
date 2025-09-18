"""Google Vertex AI provider implementation with Gemini models."""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.preview.generative_models import ChatSession
import google.auth
from google.auth import exceptions as auth_exceptions

# Configure logging
logger = logging.getLogger(__name__)

class VertexAIProvider:
    """Google Vertex AI provider using Gemini models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.project_id = self.config.get('project_id') or os.getenv('VERTEX_AI_PROJECT_ID')
        self.region = self.config.get('region', 'us-central1')
        self.model_name = self.config.get('model', 'gemini-1.5-flash')
        
        if not self.project_id:
            raise ValueError("VERTEX_AI_PROJECT_ID environment variable or project_id config required")
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.region)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Vertex AI with project {self.project_id} in {self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Vertex AI Gemini."""
        try:
            # Convert messages to Vertex AI format
            if not messages:
                raise ValueError("No messages provided")
            
            # System message handling
            system_message = None
            chat_messages = []
            
            for msg in messages:
                if msg.get('role') == 'system':
                    system_message = msg['content']
                elif msg.get('role') in ['user', 'model']:
                    # Vertex AI uses 'model' instead of 'assistant'
                    role = 'model' if msg.get('role') == 'assistant' else msg.get('role')
                    chat_messages.append({
                        'role': role,
                        'parts': [Part.from_text(msg['content'])]
                    })
            
            # If no system message, add educational context
            if not system_message:
                system_message = (
                    "Jesteś AI tutorem dla Minecraft Education. "
                    "Pomagasz uczniom uczyć się programowania, matematyki, fizyki "
                    "i rozwiązywania problemów przez Minecraft. "
                    "Bądź zachęcający, jasny i edukacyjny. "
                    "Odpowiadaj w języku polskim."
                )
            
            # Create generation config
            generation_config = {
                'temperature': kwargs.get('temperature', 0.7),
                'max_output_tokens': kwargs.get('max_tokens', 1000),
                'top_p': kwargs.get('top_p', 0.95),
                'top_k': kwargs.get('top_k', 40)
            }
            
            # For single message, use generate_content
            if len(chat_messages) == 1 and chat_messages[0]['role'] == 'user':
                prompt = chat_messages[0]['parts'][0].text
                if system_message:
                    prompt = f"System: {system_message}\n\nUser: {prompt}"
                
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=generation_config
                )
                
                return {
                    'content': response.text,
                    'provider': 'vertex',
                    'model': self.model_name,
                    'usage': {
                        'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                        'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                        'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                    } if hasattr(response, 'usage_metadata') else None
                }
            
            # For multi-turn conversation, use chat session
            else:
                chat = self.model.start_chat()
                
                # Add system instruction if available
                if system_message:
                    system_response = await asyncio.to_thread(
                        chat.send_message,
                        f"System instruction: {system_message}. Acknowledge this instruction briefly."
                    )
                
                # Process chat messages
                for msg in chat_messages:
                    if msg['role'] == 'user':
                        response = await asyncio.to_thread(
                            chat.send_message,
                            msg['parts'][0].text
                        )
                
                return {
                    'content': response.text,
                    'provider': 'vertex',
                    'model': self.model_name,
                    'usage': {
                        'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                        'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                        'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                    } if hasattr(response, 'usage_metadata') else None
                }
                
        except Exception as e:
            logger.error(f"Vertex AI API error: {e}")
            raise RuntimeError(f"Vertex AI generation failed: {str(e)}")
    
    async def is_available(self) -> bool:
        """Check if Vertex AI is available and properly configured."""
        try:
            # Test authentication
            credentials, project = google.auth.default()
            
            if not self.project_id:
                return False
            
            # Test with simple generation
            test_response = await asyncio.to_thread(
                self.model.generate_content,
                "Test",
                generation_config={'max_output_tokens': 10}
            )
            
            return bool(test_response.text)
            
        except auth_exceptions.DefaultCredentialsError:
            logger.warning("Google Cloud credentials not found")
            return False
        except Exception as e:
            logger.warning(f"Vertex AI provider not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'provider': 'vertex',
            'model': self.model_name,
            'project_id': self.project_id,
            'region': self.region,
            'capabilities': [
                'text_generation',
                'chat',
                'multilingual',
                'code_generation',
                'reasoning'
            ]
        }

class VertexAIEmbeddingProvider:
    """Vertex AI Text Embeddings provider."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.project_id = self.config.get('project_id') or os.getenv('VERTEX_AI_PROJECT_ID')
        self.region = self.config.get('region', 'us-central1')
        self.model_name = self.config.get('embedding_model', 'text-embedding-004')
        
        if not self.project_id:
            raise ValueError("VERTEX_AI_PROJECT_ID required for embeddings")
        
        try:
            vertexai.init(project=self.project_id, location=self.region)
            from vertexai.language_models import TextEmbeddingModel
            self.embedding_model = TextEmbeddingModel.from_pretrained(self.model_name)
            logger.info(f"Initialized Vertex AI embeddings with {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI embeddings: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            # Batch embedding generation
            embeddings = await asyncio.to_thread(
                self.embedding_model.get_embeddings,
                texts
            )
            
            return [embedding.values for embedding in embeddings]
            
        except Exception as e:
            logger.error(f"Vertex AI embedding error: {e}")
            raise RuntimeError(f"Vertex AI embedding failed: {str(e)}")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []

# Factory function for easy integration
def create_vertex_ai_provider(config: Dict[str, Any] = None) -> VertexAIProvider:
    """Create and return a configured Vertex AI provider."""
    return VertexAIProvider(config)

def create_vertex_embedding_provider(config: Dict[str, Any] = None) -> VertexAIEmbeddingProvider:
    """Create and return a configured Vertex AI embedding provider."""
    return VertexAIEmbeddingProvider(config)

# Test function
async def test_vertex_ai():
    """Test Vertex AI connectivity and functionality."""
    try:
        provider = create_vertex_ai_provider()
        
        # Test availability
        if not await provider.is_available():
            print("❌ Vertex AI provider is not available")
            return False
        
        print("✅ Vertex AI provider is available")
        
        # Test generation
        messages = [
            {
                'role': 'user',
                'content': 'Jak mogę nauczyć się programowania w Minecraft Education?'
            }
        ]
        
        response = await provider.generate_response(messages)
        print(f"✅ Test response: {response['content'][:100]}...")
        
        # Test embedding
        embedding_provider = create_vertex_embedding_provider()
        embeddings = await embedding_provider.embed_text("Test embedding")
        print(f"✅ Embedding dimension: {len(embeddings)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    asyncio.run(test_vertex_ai())