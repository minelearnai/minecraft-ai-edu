from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Optional, List
from ai import AIProviderFactory, AIConfig, ChatMessage, MessageRole, ChatResponse
from search import find_relevant_docs, initialize_search

class ChatRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[str] = None
    provider: Optional[str] = None  # Allow user to specify provider

class ChatResponseModel(BaseModel):
    response: str
    sources: List[str] = []
    user_id: str
    provider: str
    model: str
    response_time_ms: int

# Initialize FastAPI app
app = FastAPI(
    title="Minecraft AI Edu Chatbot",
    description="AI-powered educational assistant for Minecraft Education Edition with multiple AI providers",
    version="2.0.0"
)

# Global variables
ai_factory: Optional[AIProviderFactory] = None
ai_config: Optional[AIConfig] = None

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    global ai_factory, ai_config
    
    try:
        # Initialize AI configuration from environment
        ai_config = AIConfig.from_env()
        print(f"🤖 AI Config loaded - Default provider: {ai_config.default_provider}")
        
        # Initialize AI factory
        ai_factory = AIProviderFactory(ai_config)
        print("🏭 AI Provider Factory initialized")
        
        # Test default provider
        try:
            default_provider = await ai_factory.get_provider()
            model_info = default_provider.get_model_info()
            print(f"✅ Default AI Provider ready: {model_info['provider']} ({model_info['model']})")
        except Exception as e:
            print(f"⚠️ Default AI provider test failed: {e}")
        
        # Initialize search service
        try:
            await initialize_search()
            print("🔍 Search service initialized")
        except Exception as e:
            print(f"⚠️ Search service failed: {e}")
            
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Minecraft AI Edu Backend v2.0", 
        "status": "running",
        "features": ["Multi-AI-Provider", "Educational-Context", "Minecraft-Focused"]
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "backend", "version": "2.0.0"}

@app.post("/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequest):
    global ai_factory
    
    if not ai_factory:
        raise HTTPException(status_code=503, detail="AI services not initialized")
    
    try:
        # Get AI provider (use requested provider or default)
        provider_name = None
        if request.provider:
            from ai.config import AIProvider
            try:
                provider_name = AIProvider(request.provider.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid provider: {request.provider}")
        
        provider = await ai_factory.get_provider(provider_name)
        
        # Find relevant documents from knowledge base
        docs = []
        try:
            docs = await find_relevant_docs(request.message)
        except Exception as e:
            print(f"⚠️ Knowledge search failed: {e}")
        
        # Build educational context
        context_parts = []
        
        if ai_config.educational_context:
            context_parts.append(
                "You are an educational AI assistant specialized in teaching through Minecraft Education Edition. "
                "Your role is to help students aged 10-15 learn mathematics, physics, and other STEM subjects "
                "using Minecraft as an interactive learning environment."
            )
        
        if ai_config.minecraft_focused:
            context_parts.append(
                "Always relate your answers to Minecraft when possible. Use Minecraft blocks, tools, and concepts "
                "as examples. Suggest hands-on Minecraft activities that reinforce the learning concepts."
            )
        
        if docs:
            context_parts.append(f"Relevant educational content:\n{chr(10).join(docs[:3])}")
        elif request.context:
            context_parts.append(f"Additional context: {request.context}")
        
        # Create messages for AI provider
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=" ".join(context_parts) if context_parts else "You are a helpful educational assistant."
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=request.message
            )
        ]
        
        # Generate AI response
        response = await provider.generate_response(
            messages=messages,
            max_tokens=ai_config.max_tokens,
            temperature=ai_config.temperature
        )
        
        return ChatResponseModel(
            response=response.message.content,
            sources=[doc[:100] + "..." for doc in docs[:3]] if docs else [],
            user_id=request.user_id,
            provider=response.provider,
            model=response.model,
            response_time_ms=response.response_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Streaming chat endpoint"""
    global ai_factory
    
    if not ai_factory:
        raise HTTPException(status_code=503, detail="AI services not initialized")
    
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate_stream():
        try:
            # Get provider
            provider_name = None
            if request.provider:
                from ai.config import AIProvider
                provider_name = AIProvider(request.provider.lower())
            
            provider = await ai_factory.get_provider(provider_name)
            
            if not provider.supports_streaming():
                yield f"data: {{\"error\": \"Provider {provider.provider_name} doesn't support streaming\"}}\n\n"
                return
            
            # Find context (simplified for streaming)
            docs = []
            try:
                docs = await find_relevant_docs(request.message)
            except:
                pass
            
            # Build messages
            context = "You are an educational AI assistant for Minecraft Education Edition."
            if docs:
                context += f" Relevant info: {docs[0][:200]}..."
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=context),
                ChatMessage(role=MessageRole.USER, content=request.message)
            ]
            
            # Stream response
            async for chunk in provider.stream_response(
                messages=messages,
                max_tokens=ai_config.max_tokens,
                temperature=ai_config.temperature
            ):
                yield f"data: {{\"content\": \"{chunk.replace('"', '\\"')}\", \"type\": \"chunk\"}}\n\n"
            
            # Send completion signal
            yield f"data: {{\"type\": \"done\", \"provider\": \"{provider.provider_name}\"}}\n\n"
            
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/ai/providers")
async def list_providers():
    """List available AI providers and their status"""
    global ai_factory
    
    if not ai_factory:
        raise HTTPException(status_code=503, detail="AI services not initialized")
    
    try:
        status = ai_factory.get_provider_status()
        available = await ai_factory.get_all_available_providers()
        
        return {
            "default_provider": ai_config.default_provider.value,
            "available_providers": [p.value for p in available],
            "provider_status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/test")
async def test_providers():
    """Test all configured AI providers"""
    global ai_factory
    
    if not ai_factory:
        raise HTTPException(status_code=503, detail="AI services not initialized")
    
    try:
        results = await ai_factory.test_all_providers()
        return {"test_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/status")
async def knowledge_status():
    """Check knowledge base status"""
    try:
        from search import get_document_count
        count = await get_document_count()
        return {"status": "ok", "document_count": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/config")
async def get_config():
    """Get current AI configuration (sensitive data removed)"""
    global ai_config
    
    if not ai_config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    return {
        "default_provider": ai_config.default_provider.value,
        "fallback_providers": [p.value for p in ai_config.fallback_providers],
        "educational_context": ai_config.educational_context,
        "minecraft_focused": ai_config.minecraft_focused,
        "max_tokens": ai_config.max_tokens,
        "temperature": ai_config.temperature,
        "models": {
            "openai": ai_config.openai_model,
            "local": ai_config.local_model,
            "anthropic": ai_config.anthropic_model
        }
    }