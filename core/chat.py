import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator

from loguru import logger

from core.config import settings
from core.database import db_manager
from core.rag import rag_engine, temporal_engine, hallucination_reducer
from core.llm_service import llm_service


class ChatSession:
    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.messages: List[Dict] = []
        self.context_window = 10  # Number of previous messages to consider

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None, 
                   parent_message_id: Optional[str] = None, thread_id: Optional[str] = None):
        message = {
            "id": str(uuid.uuid4()),
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "parent_message_id": parent_message_id,
            "thread_id": thread_id or str(uuid.uuid4())
        }
        self.messages.append(message)
        return message

    def get_recent_messages(self, limit: Optional[int] = None) -> List[Dict]:
        limit = limit or self.context_window
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def get_conversation_context(self) -> str:
        recent_messages = self.get_recent_messages()
        context_parts = []
        
        for msg in recent_messages:
            role = msg["role"].upper()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages),
            "messages": self.messages
        }


class ChatManager:
    def __init__(self):
        self.active_sessions: Dict[str, ChatSession] = {}

    def create_session(self, user_id: Optional[str] = None) -> ChatSession:
        session = ChatSession(user_id=user_id)
        self.active_sessions[session.session_id] = session
        logger.info(f"Created new chat session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.active_sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            logger.info(f"Ended chat session: {session_id}")
            return True
        return False

    async def save_session_to_redis(self, session: ChatSession):
        """Save session data to Redis for persistence"""
        try:
            redis_client = await db_manager.get_redis_client()
            session_key = f"chat_session:{session.session_id}"
            session_data = session.to_dict()
            
            await redis_client.setex(
                session_key,
                3600 * 24,  # 24 hours TTL
                str(session_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to save session to Redis: {e}")

    async def load_session_from_redis(self, session_id: str) -> Optional[ChatSession]:
        """Load session data from Redis"""
        try:
            redis_client = await db_manager.get_redis_client()
            session_key = f"chat_session:{session_id}"
            session_data = await redis_client.get(session_key)
            
            if session_data:
                # Parse session data and recreate ChatSession
                # This is a simplified implementation
                session = ChatSession(session_id=session_id)
                self.active_sessions[session_id] = session
                return session
                
        except Exception as e:
            logger.error(f"Failed to load session from Redis: {e}")
        
        return None


class ChatEngine:
    def __init__(self):
        self.chat_manager = ChatManager()

    async def process_message(self, 
                            session_id: str, 
                            user_message: str,
                            use_rag: bool = True,
                            rag_limit: Optional[int] = None,
                            parent_message_id: Optional[str] = None,
                            thread_id: Optional[str] = None) -> Dict:
        try:
            session = self.chat_manager.get_session(session_id)
            if not session:
                session = self.chat_manager.create_session()
            
            # Add user message to session with threading support
            user_msg = session.add_message("user", user_message, 
                                          parent_message_id=parent_message_id, 
                                          thread_id=thread_id)
            
            # Get conversation context
            conversation_context = session.get_conversation_context()
            
            if use_rag:
                # Use RAG to generate response
                rag_response = await rag_engine.query(user_message, rag_limit)
                
                # Add temporal awareness
                if "context_chunks" in rag_response and rag_response["context_chunks"]:
                    # This would be implemented with actual context chunks
                    pass
                
                # Validate response for hallucinations
                validation = await hallucination_reducer.validate_response(
                    rag_response.get("response", ""), 
                    []  # Would pass actual context chunks
                )
                
                assistant_content = rag_response.get("response", "No response generated")
                response_metadata = {
                    "rag_used": True,
                    "context_chunks": rag_response.get("context_chunks", 0),
                    "rag_metadata": rag_response.get("metadata", {}),
                    "validation": validation
                }
            else:
                # Response without RAG - use LLM if available
                if settings.enable_llm_chat and await llm_service.is_available():
                    # Use LLM without RAG context
                    try:
                        assistant_content = await llm_service.generate_response(user_message, [])
                        response_metadata = {
                            "rag_used": False,
                            "context_chunks": 0,
                            "generation_method": "llm_without_rag",
                            "llm_info": llm_service.get_provider_info()
                        }
                    except Exception as llm_error:
                        logger.warning(f"LLM generation failed: {llm_error}")
                        assistant_content = "I apologize, but I'm having trouble generating a response right now. Please try again or enable RAG for document-based answers."
                        response_metadata = {
                            "rag_used": False,
                            "context_chunks": 0,
                            "error": str(llm_error)
                        }
                else:
                    # LLM not available - provide informative fallback
                    llm_status = "disabled" if not settings.enable_llm_chat else "unavailable"
                    assistant_content = f"I can help you, but the LLM service is currently {llm_status}. Please enable RAG for document-based responses, or check the LLM configuration."
                    response_metadata = {
                        "rag_used": False,
                        "context_chunks": 0,
                        "llm_status": llm_status
                    }
            
            # Add assistant response to session with threading support
            assistant_msg = session.add_message("assistant", assistant_content, response_metadata,
                                               thread_id=user_msg.get("thread_id"))
            
            # Save session to Redis
            await self.chat_manager.save_session_to_redis(session)
            
            response = {
                "session_id": session.session_id,
                "user_message": user_msg,
                "assistant_message": assistant_msg,
                "conversation_length": len(session.messages),
                "rag_used": use_rag,
                "metadata": response_metadata
            }
            
            logger.info(f"Processed message in session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise

    async def process_message_stream(self, 
                                   session_id: str, 
                                   user_message: str,
                                   use_rag: bool = True,
                                   rag_limit: Optional[int] = None,
                                   parent_message_id: Optional[str] = None,
                                   thread_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Process message with streaming response using Server-Sent Events format"""
        try:
            session = self.chat_manager.get_session(session_id)
            if not session:
                session = self.chat_manager.create_session()
            
            # Add user message to session with threading support
            user_msg = session.add_message("user", user_message, 
                                          parent_message_id=parent_message_id, 
                                          thread_id=thread_id)
            
            # Yield initial message info
            initial_chunk = {
                "type": "message_start",
                "session_id": session.session_id,
                "user_message": user_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
            yield json.dumps(initial_chunk)
            
            # Get conversation context
            conversation_context = session.get_conversation_context()
            
            # Initialize response variables
            assistant_content = ""
            response_metadata = {}
            
            if use_rag:
                # Yield RAG retrieval start
                rag_start_chunk = {
                    "type": "rag_start", 
                    "message": "Retrieving relevant context..."
                }
                yield json.dumps(rag_start_chunk)
                
                # Get context chunks only (not the response)
                context_chunks = await rag_engine.retrieve_context(user_message, rag_limit)
                
                # Yield context found
                context_chunk = {
                    "type": "context_found",
                    "context_chunks": len(context_chunks),
                    "metadata": {"retrieval_count": len(context_chunks)}
                }
                yield json.dumps(context_chunk)
                
                # Check if LLM is available for streaming
                if settings.enable_llm_chat and await llm_service.is_available():
                    # Use LLM streaming for response generation
                    generation_chunk = {
                        "type": "generation_start",
                        "message": f"Generating response using {settings.llm_provider}..."
                    }
                    yield json.dumps(generation_chunk)
                    
                    accumulated_text = ""
                    try:
                        async for chunk_text in llm_service.stream_response(user_message, context_chunks):
                            accumulated_text += chunk_text
                            text_chunk = {
                                "type": "text_chunk",
                                "content": chunk_text,
                                "accumulated_text": accumulated_text
                            }
                            yield json.dumps(text_chunk)
                        
                        assistant_content = accumulated_text
                        response_metadata = {
                            "rag_used": True,
                            "context_chunks": len(context_chunks),
                            "generation_method": "llm_streaming",
                            "llm_info": llm_service.get_provider_info()
                        }
                        
                    except Exception as llm_error:
                        logger.warning(f"LLM streaming failed: {llm_error}")
                        if settings.llm_fallback_to_extractive:
                            # Fallback to extractive with context chunks
                            fallback_chunk = {
                                "type": "fallback_start",
                                "message": "LLM unavailable, using extractive summarization..."
                            }
                            yield json.dumps(fallback_chunk)
                            
                            assistant_content = await rag_engine._generate_extractive_response(user_message, context_chunks)
                            response_metadata = {
                                "rag_used": True,
                                "context_chunks": len(context_chunks),
                                "generation_method": "extractive_fallback"
                            }
                            
                            # Stream extractive response
                            text_chunk = {
                                "type": "text_chunk",
                                "content": assistant_content,
                                "accumulated_text": assistant_content
                            }
                            yield json.dumps(text_chunk)
                        else:
                            raise llm_error
                else:
                    # Use extractive summarization with context chunks
                    extractive_chunk = {
                        "type": "generation_start",
                        "message": "Generating response using extractive summarization..."
                    }
                    yield json.dumps(extractive_chunk)
                    
                    assistant_content = await rag_engine._generate_extractive_response(user_message, context_chunks)
                    response_metadata = {
                        "rag_used": True,
                        "context_chunks": len(context_chunks),
                        "generation_method": "extractive_summarization"
                    }
                    
                    # Stream extractive response
                    text_chunk = {
                        "type": "text_chunk",
                        "content": assistant_content,
                        "accumulated_text": assistant_content
                    }
                    yield json.dumps(text_chunk)
            else:
                # Response without RAG - use LLM streaming if available
                if settings.enable_llm_chat and await llm_service.is_available():
                    # Use LLM streaming without RAG context
                    try:
                        generation_chunk = {
                            "type": "generation_start",
                            "message": f"Generating response using {settings.llm_provider} (no RAG)..."
                        }
                        yield json.dumps(generation_chunk)
                        
                        accumulated_text = ""
                        async for chunk_text in llm_service.stream_response(user_message, []):
                            accumulated_text += chunk_text
                            text_chunk = {
                                "type": "text_chunk",
                                "content": chunk_text,
                                "accumulated_text": accumulated_text
                            }
                            yield json.dumps(text_chunk)
                        
                        assistant_content = accumulated_text
                        response_metadata = {
                            "rag_used": False,
                            "context_chunks": 0,
                            "generation_method": "llm_streaming_without_rag",
                            "llm_info": llm_service.get_provider_info()
                        }
                        
                    except Exception as llm_error:
                        logger.warning(f"LLM streaming failed: {llm_error}")
                        assistant_content = "I apologize, but I'm having trouble generating a response right now. Please try again or enable RAG for document-based answers."
                        response_metadata = {
                            "rag_used": False,
                            "context_chunks": 0,
                            "error": str(llm_error)
                        }
                        
                        # Stream error response
                        text_chunk = {
                            "type": "text_chunk",
                            "content": assistant_content,
                            "accumulated_text": assistant_content,
                            "progress": 1.0
                        }
                        yield json.dumps(text_chunk)
                else:
                    # LLM not available - provide informative fallback
                    llm_status = "disabled" if not settings.enable_llm_chat else "unavailable"
                    assistant_content = f"I can help you, but the LLM service is currently {llm_status}. Please enable RAG for document-based responses, or check the LLM configuration."
                    response_metadata = {
                        "rag_used": False,
                        "context_chunks": 0,
                        "llm_status": llm_status
                    }
                    
                    # Stream fallback response
                    text_chunk = {
                        "type": "text_chunk",
                        "content": assistant_content,
                        "accumulated_text": assistant_content,
                        "progress": 1.0
                    }
                    yield json.dumps(text_chunk)
            
            # Add assistant response to session
            assistant_msg = session.add_message("assistant", assistant_content, response_metadata,
                                               thread_id=user_msg.get("thread_id"))
            
            # Save session to Redis
            await self.chat_manager.save_session_to_redis(session)
            
            # Yield final completion
            final_chunk = {
                "type": "message_complete",
                "session_id": session.session_id,
                "assistant_message": assistant_msg,
                "conversation_length": len(session.messages),
                "metadata": response_metadata
            }
            yield json.dumps(final_chunk)
            
            logger.info(f"Completed streaming message in session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to process streaming message: {e}")
            error_chunk = {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            yield json.dumps(error_chunk)

    async def get_session_history(self, session_id: str) -> Optional[Dict]:
        session = self.chat_manager.get_session(session_id)
        if not session:
            session = await self.chat_manager.load_session_from_redis(session_id)
        
        return session.to_dict() if session else None

    async def create_new_session(self, user_id: Optional[str] = None) -> str:
        session = self.chat_manager.create_session(user_id)
        await self.chat_manager.save_session_to_redis(session)
        return session.session_id

    async def end_session(self, session_id: str) -> bool:
        return self.chat_manager.end_session(session_id)

    def get_active_sessions(self) -> List[str]:
        return list(self.chat_manager.active_sessions.keys())


chat_engine = ChatEngine()