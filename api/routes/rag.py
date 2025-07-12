from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.rag import rag_engine
from core.indonesian_kg_manager import indonesian_kg_manager

router = APIRouter()


class RAGQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = None
    use_llm: Optional[bool] = False  # Default to False (no LLM)
    extractive_only: Optional[bool] = False  # Force extractive mode


class RAGQueryResponse(BaseModel):
    query: str
    response: str
    context_chunks: int
    metadata: Dict
    language: Optional[str] = None
    indonesian_context: Optional[Dict] = None


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Process a RAG query and return a response with context.
    
    Parameters:
    - query: The question or search query
    - limit: Maximum number of context chunks to retrieve (optional)
    - use_llm: Override default LLM setting (true/false, optional)
    - extractive_only: Force extractive summarization mode (default: false)
    
    LLM Control Options:
    - extractive_only=true: Always use extractive summarization 
    - use_llm=false: Disable LLM for this request
    - use_llm=true: Enable LLM for this request (if available)
    - No parameters: Use system default configuration
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Determine LLM usage based on request parameters
        llm_options = {}
        if request.extractive_only:
            llm_options["force_extractive"] = True
        elif request.use_llm is not None:
            llm_options["use_llm"] = request.use_llm
        
        result = await rag_engine.query(
            request.query, 
            request.limit, 
            **llm_options
        )
        
        return RAGQueryResponse(
            query=result["query"],
            response=result["response"],
            context_chunks=result.get("context_chunks", 0),
            metadata=result.get("metadata", {}),
            language=result.get("language"),
            indonesian_context=result.get("indonesian_context")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/rag/retrieve")
async def retrieve_context(query: str, limit: int = 10):
    """Retrieve relevant context for a query without generating a response"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        context_chunks = await rag_engine.retrieve_context(query, limit)
        
        return {
            "query": query,
            "context_chunks": context_chunks,
            "count": len(context_chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")


@router.get("/rag/config")
async def get_rag_config():
    """Get current RAG configuration"""
    try:
        from core.config import settings
        
        # Get Indonesian KG metrics if available
        indonesian_metrics = {}
        try:
            metrics = await indonesian_kg_manager.get_comprehensive_kg_metrics()
            indonesian_metrics = {
                "total_entities": metrics.get('derived_metrics', {}).get('total_entities', 0),
                "quality_score": metrics.get('derived_metrics', {}).get('quality_score', 0),
                "high_confidence_ratio": metrics.get('derived_metrics', {}).get('high_confidence_ratio', 0)
            }
        except Exception:
            pass
        
        return {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
            "similarity_threshold": settings.similarity_threshold,
            "embedding_model": settings.embedding_model,
            "extraction_model": settings.extraction_model,
            "indonesian_kg_metrics": indonesian_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


