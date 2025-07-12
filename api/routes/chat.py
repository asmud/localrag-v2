from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.chat import chat_engine
from core.postgres_storage import chat_storage

router = APIRouter()


class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: bool = True
    rag_limit: Optional[int] = None


class ChatMessageResponse(BaseModel):
    session_id: str
    user_message: Dict
    assistant_message: Dict
    conversation_length: int
    rag_used: bool
    metadata: Dict


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str]
    created_at: str


class SaveSessionRequest(BaseModel):
    metadata: Optional[Dict] = None


class SaveSessionResponse(BaseModel):
    success: bool
    message: str
    saved_messages: int


class ThreadCreateRequest(BaseModel):
    parent_message_id: str
    new_thread_id: Optional[str] = None


class ThreadCreateResponse(BaseModel):
    thread_id: str
    parent_message_id: str
    session_id: str


class ChatStreamRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: bool = True
    rag_limit: Optional[int] = None
    parent_message_id: Optional[str] = None
    thread_id: Optional[str] = None


@router.post("/chat/sessions", response_model=CreateSessionResponse)
async def create_chat_session(request: CreateSessionRequest):
    """Create a new chat session"""
    try:
        session_id = await chat_engine.create_new_session(request.user_id)
        
        return CreateSessionResponse(
            session_id=session_id,
            user_id=request.user_id,
            created_at="now"  # Would be actual timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.post("/chat/message", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest):
    """Send a message in a chat session"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Create new session if none provided
        session_id = request.session_id
        if not session_id:
            session_id = await chat_engine.create_new_session()
        
        result = await chat_engine.process_message(
            session_id=session_id,
            user_message=request.message,
            use_rag=request.use_rag,
            rag_limit=request.rag_limit
        )
        
        return ChatMessageResponse(
            session_id=result["session_id"],
            user_message=result["user_message"],
            assistant_message=result["assistant_message"],
            conversation_length=result["conversation_length"],
            rag_used=result["rag_used"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get chat session history"""
    try:
        session_data = await chat_engine.get_session_history(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.delete("/chat/sessions/{session_id}")
async def end_chat_session(session_id: str):
    """End a chat session"""
    try:
        success = await chat_engine.end_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": f"Session {session_id} ended successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.get("/chat/sessions")
async def list_active_sessions():
    """List all active chat sessions"""
    try:
        active_sessions = chat_engine.get_active_sessions()
        
        return {
            "active_sessions": active_sessions,
            "count": len(active_sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


# Chat Persistence Endpoints

@router.post("/chat/sessions/{session_id}/save", response_model=SaveSessionResponse)
async def save_chat_session_to_postgres(session_id: str, request: SaveSessionRequest):
    """Save chat session and messages to PostgreSQL for persistence"""
    try:
        # Get session from memory/Redis
        session_data = await chat_engine.get_session_history(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found in memory")
        
        # Save session metadata to PostgreSQL
        session_saved = await chat_storage.save_chat_session(
            session_id=session_id,
            user_id=session_data.get('user_id'),
            metadata=request.metadata
        )
        
        if not session_saved:
            raise HTTPException(status_code=500, detail="Failed to save session metadata")
        
        # Save messages to PostgreSQL
        messages = session_data.get('messages', [])
        messages_saved = False
        saved_count = 0
        
        if messages:
            messages_saved = await chat_storage.save_chat_messages(session_id, messages)
            saved_count = len(messages)
        
        if not messages_saved and messages:
            raise HTTPException(status_code=500, detail="Failed to save messages")
        
        return SaveSessionResponse(
            success=True,
            message=f"Successfully saved session {session_id} to PostgreSQL",
            saved_messages=saved_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")


@router.get("/chat/sessions/{session_id}/history")
async def get_chat_session_from_postgres(session_id: str):
    """Retrieve chat session history from PostgreSQL"""
    try:
        session_data = await chat_storage.load_chat_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found in PostgreSQL")
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")


# Message Threading Endpoints

@router.post("/chat/sessions/{session_id}/threads", response_model=ThreadCreateResponse)
async def create_message_thread(session_id: str, request: ThreadCreateRequest):
    """Create a new conversation thread from a parent message"""
    try:
        thread_id = await chat_storage.create_message_thread(
            session_id=session_id,
            parent_message_id=request.parent_message_id,
            new_thread_id=request.new_thread_id
        )
        
        if not thread_id:
            raise HTTPException(status_code=404, detail="Parent message not found or thread creation failed")
        
        return ThreadCreateResponse(
            thread_id=thread_id,
            parent_message_id=request.parent_message_id,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create thread: {str(e)}")


@router.get("/chat/sessions/{session_id}/threads")
async def list_session_threads(session_id: str):
    """List all conversation threads in a session"""
    try:
        threads = await chat_storage.get_session_threads(session_id)
        
        return {
            "session_id": session_id,
            "threads": threads,
            "count": len(threads)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list threads: {str(e)}")


@router.get("/chat/sessions/{session_id}/threads/{thread_id}")
async def get_thread_messages(session_id: str, thread_id: str, include_children: bool = True):
    """Get all messages in a specific thread"""
    try:
        messages = await chat_storage.get_thread_messages(
            session_id=session_id,
            thread_id=thread_id,
            include_children=include_children
        )
        
        return {
            "session_id": session_id,
            "thread_id": thread_id,
            "messages": messages,
            "count": len(messages)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get thread messages: {str(e)}")


# Streaming Chat Endpoint

@router.post("/chat/message/stream")
async def send_chat_message_stream(request: ChatStreamRequest):
    """Send a chat message with streaming response"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Create new session if none provided
        session_id = request.session_id
        if not session_id:
            session_id = await chat_engine.create_new_session()
        
        async def generate_stream():
            """Generate SSE stream for chat response"""
            try:
                async for chunk in chat_engine.process_message_stream(
                    session_id=session_id,
                    user_message=request.message,
                    use_rag=request.use_rag,
                    rag_limit=request.rag_limit,
                    parent_message_id=request.parent_message_id,
                    thread_id=request.thread_id
                ):
                    yield f"data: {chunk}\n\n"
                
                # Send final completion signal
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_chunk = f'{{"error": "Stream error: {str(e)}"}}'
                yield f"data: {error_chunk}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")


