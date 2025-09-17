from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Optional
from .ai_provider import get_ai_provider
from .search import find_relevant_docs, initialize_search

class ChatRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []
    user_id: str

# Initialize FastAPI app
app = FastAPI(
    title="Minecraft AI Edu Chatbot",
    description="AI-powered educational assistant for Minecraft Education Edition",
    version="1.0.0"
)

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    # Initialize search service
    await initialize_search()
    print("✅ Backend services initialized")

@app.get("/")
async def root():
    return {"message": "Minecraft AI Edu Backend", "status": "running"}

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "backend"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get AI provider instance
        provider = get_ai_provider()
        
        # Find relevant documents from knowledge base
        docs = await find_relevant_docs(request.message)
        
        # Build context-aware prompt
        context = "\n\n".join(docs) if docs else "No specific context available."
        prompt = f"Context:\n{context}\n\nStudent question: {request.message}\n\nProvide a helpful, educational response suitable for students aged 10-15."
        
        # Generate AI response
        response = await provider.generate_response(request.user_id, prompt)
        
        return ChatResponse(
            response=response,
            sources=[doc[:100] + "..." for doc in docs[:3]] if docs else [],
            user_id=request.user_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/knowledge/status")
async def knowledge_status():
    """Check knowledge base status"""
    try:
        from .search import get_document_count
        count = await get_document_count()
        return {"status": "ok", "document_count": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}