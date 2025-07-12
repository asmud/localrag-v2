"""
PostgreSQL storage service for documents and vector embeddings.
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import select, func, text, and_, or_
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError

from .database import db_manager
from .models_orm import Document, Chunk, ChatSession, ChatMessage

logger = logging.getLogger(__name__)


class PostgreSQLStorageService:
    """Service for storing and retrieving documents and chunks with vector embeddings."""
    
    def __init__(self):
        self.db_manager = db_manager
    
    async def check_document_exists_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if document with given content hash already exists.
        
        Args:
            content_hash: SHA256 hash of the document content
            
        Returns:
            Document info if exists, None otherwise
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(Document).where(Document.content_hash == content_hash)
                existing_doc = await session.scalar(stmt)
                
                if existing_doc:
                    return {
                        "id": existing_doc.id,
                        "file_path": existing_doc.file_path,
                        "file_name": existing_doc.file_name,
                        "file_size": existing_doc.file_size,
                        "content_hash": existing_doc.content_hash,
                        "created_at": existing_doc.created_at,
                        "chunk_count": existing_doc.chunk_count
                    }
                return None
                
            except Exception as e:
                logger.error(f"Error checking document by hash {content_hash[:8]}...: {e}")
                return None

    async def check_successfully_processed_document_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if document with given content hash exists AND was successfully processed.
        
        Args:
            content_hash: SHA256 hash of the document content
            
        Returns:
            Document info if exists and was successfully processed, None otherwise
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                # First check if document exists
                stmt = select(Document).where(Document.content_hash == content_hash)
                existing_doc = await session.scalar(stmt)
                
                if not existing_doc:
                    return None
                
                # Check if document has chunks (indicating successful processing)
                if existing_doc.chunk_count > 0:
                    return {
                        "id": existing_doc.id,
                        "file_path": existing_doc.file_path,
                        "file_name": existing_doc.file_name,
                        "file_size": existing_doc.file_size,
                        "content_hash": existing_doc.content_hash,
                        "created_at": existing_doc.created_at,
                        "chunk_count": existing_doc.chunk_count
                    }
                
                # If document exists but has no chunks, it wasn't successfully processed
                return None
                
            except Exception as e:
                logger.error(f"Error checking successfully processed document by hash {content_hash[:8]}...: {e}")
                return None
    
    async def store_document(
        self, 
        file_path: str, 
        file_name: str, 
        file_size: int,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store document metadata and return document ID.
        
        Args:
            file_path: Path to the document file
            file_name: Name of the document file
            file_size: Size of the document in bytes
            content_hash: SHA256 hash of the document content
            metadata: Additional metadata as JSON
            
        Returns:
            Document ID
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                # Check if document already exists by file path
                stmt = select(Document).where(Document.file_path == file_path)
                existing_doc = await session.scalar(stmt)
                
                if existing_doc:
                    logger.info(f"Document already exists at path: {file_path}")
                    return existing_doc.id
                
                # Check if document with same content hash already exists
                if content_hash:
                    stmt = select(Document).where(Document.content_hash == content_hash)
                    existing_doc_by_hash = await session.scalar(stmt)
                    
                    if existing_doc_by_hash:
                        logger.info(f"Document with identical content already exists: {existing_doc_by_hash.file_path} (content hash: {content_hash[:8]}...)")
                        return existing_doc_by_hash.id
                
                # Create new document
                document = Document(
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size,
                    content_hash=content_hash,
                    chunk_count=0,
                    doc_meta=metadata or {}
                )
                
                session.add(document)
                await session.flush()  # Get the ID without committing
                
                logger.info(f"Stored document: {file_name} (ID: {document.id})")
                return document.id
                
            except IntegrityError as e:
                logger.error(f"Failed to store document {file_path}: {e}")
                await session.rollback()
                raise
    
    async def store_chunks_with_embeddings(
        self,
        document_id: int,
        chunks_data: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Store text chunks with their vector embeddings.
        
        Args:
            document_id: ID of the parent document
            chunks_data: List of chunk data dictionaries containing:
                - content: Text content
                - start_index: Start position in document
                - end_index: End position in document
                - word_count: Number of words
                - chunk_index: Sequential chunk number
                - embedding: Vector embedding as list of floats
                - metadata: Additional metadata
                
        Returns:
            List of chunk IDs
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                chunk_ids = []
                
                for chunk_data in chunks_data:
                    chunk = Chunk(
                        document_id=document_id,
                        content=chunk_data['content'],
                        start_index=chunk_data.get('start_index', 0),
                        end_index=chunk_data.get('end_index', len(chunk_data['content'])),
                        word_count=chunk_data.get('word_count', len(chunk_data['content'].split())),
                        chunk_index=chunk_data['chunk_index'],
                        embedding=chunk_data['embedding'],
                        doc_meta=chunk_data.get('metadata', {})
                    )
                    
                    session.add(chunk)
                    await session.flush()
                    chunk_ids.append(chunk.id)
                
                # Update document chunk count
                stmt = select(Document).where(Document.id == document_id)
                document = await session.scalar(stmt)
                if document:
                    document.chunk_count = len(chunks_data)
                
                logger.info(f"Stored {len(chunks_data)} chunks for document {document_id}")
                return chunk_ids
                
            except Exception as e:
                logger.error(f"Failed to store chunks for document {document_id}: {e}")
                await session.rollback()
                raise
    
    async def find_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.4,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find chunks similar to the query embedding using cosine similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of similar chunks with similarity scores
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                # Build query with vector similarity search
                # Using cosine similarity with pgvector
                query_vector = str(query_embedding)
                
                base_query = f"""
                SELECT 
                    c.id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    c.word_count,
                    c.metadata,
                    c.created_at,
                    d.file_name,
                    d.file_path,
                    1 - (c.embedding <=> '{query_vector}') as similarity_score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                """
                
                # Add document filter if specified
                if document_ids:
                    doc_ids_str = ','.join(map(str, document_ids))
                    base_query += f" AND c.document_id IN ({doc_ids_str})"
                
                # Add similarity threshold and ordering
                base_query += f"""
                AND (1 - (c.embedding <=> '{query_vector}')) >= {similarity_threshold}
                ORDER BY c.embedding <=> '{query_vector}'
                LIMIT {limit}
                """
                
                result = await session.execute(text(base_query))
                rows = result.fetchall()
                
                chunks = []
                for row in rows:
                    chunks.append({
                        'id': row.id,
                        'document_id': row.document_id,
                        'content': row.content,
                        'chunk_index': row.chunk_index,
                        'word_count': row.word_count,
                        'metadata': row.metadata,
                        'created_at': row.created_at.isoformat() if row.created_at else None,
                        'file_name': row.file_name,
                        'file_path': row.file_path,
                        'similarity_score': float(row.similarity_score)
                    })
                
                logger.info(f"Found {len(chunks)} similar chunks")
                return chunks
                
            except Exception as e:
                logger.error(f"Failed to find similar chunks: {e}")
                raise
    
    async def get_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get document by file path."""
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(Document).where(Document.file_path == file_path)
                document = await session.scalar(stmt)
                
                if document:
                    return document.to_dict()
                return None
                
            except Exception as e:
                logger.error(f"Failed to get document by path {file_path}: {e}")
                raise
    
    async def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
                chunks = await session.scalars(stmt)
                
                return [chunk.to_dict() for chunk in chunks]
                
            except Exception as e:
                logger.error(f"Failed to get chunks for document {document_id}: {e}")
                raise
    
    async def get_documents_count(self) -> int:
        """Get total number of documents."""
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(func.count(Document.id))
                count = await session.scalar(stmt)
                return count or 0
                
            except Exception as e:
                logger.error(f"Failed to get documents count: {e}")
                raise
    
    async def get_chunks_count(self) -> int:
        """Get total number of chunks."""
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(func.count(Chunk.id))
                count = await session.scalar(stmt)
                return count or 0
                
            except Exception as e:
                logger.error(f"Failed to get chunks count: {e}")
                raise
    
    async def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks."""
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(Document).where(Document.id == document_id)
                document = await session.scalar(stmt)
                
                if not document:
                    return False
                
                await session.delete(document)
                logger.info(f"Deleted document {document_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete document {document_id}: {e}")
                raise
    
    @staticmethod
    def calculate_content_hash(content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of raw file content for duplicate detection."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash for {file_path}: {e}")
            # Fallback to a simple hash based on file metadata
            stat = file_path.stat()
            fallback_content = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.sha256(fallback_content.encode('utf-8')).hexdigest()


class ChatStorageService:
    """Service for storing and retrieving chat sessions and messages in PostgreSQL."""
    
    def __init__(self):
        self.db_manager = db_manager
    
    async def save_chat_session(
        self, 
        session_id: str, 
        user_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save chat session to PostgreSQL.
        
        Args:
            session_id: UUID string of the session
            user_id: Optional user identifier
            metadata: Additional session metadata
            
        Returns:
            True if saved successfully
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                # Check if session already exists
                stmt = select(ChatSession).where(ChatSession.id == uuid.UUID(session_id))
                existing_session = await session.scalar(stmt)
                
                if existing_session:
                    # Update existing session
                    existing_session.user_id = user_id
                    existing_session.doc_meta = metadata or {}
                    logger.info(f"Updated existing chat session: {session_id}")
                else:
                    # Create new session
                    chat_session = ChatSession(
                        id=uuid.UUID(session_id),
                        user_id=user_id,
                        message_count=0,
                        doc_meta=metadata or {}
                    )
                    session.add(chat_session)
                    logger.info(f"Saved new chat session: {session_id}")
                
                await session.commit()
                return True
                
            except Exception as e:
                logger.error(f"Failed to save chat session {session_id}: {e}")
                await session.rollback()
                return False
    
    async def save_chat_messages(
        self, 
        session_id: str, 
        messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Save chat messages to PostgreSQL.
        
        Args:
            session_id: UUID string of the session
            messages: List of message dictionaries with keys:
                - id: message UUID
                - role: 'user' or 'assistant'
                - content: message text
                - metadata: optional metadata dict
                - parent_message_id: optional parent message UUID for threading
                - thread_id: optional thread UUID
                
        Returns:
            True if saved successfully
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                session_uuid = uuid.UUID(session_id)
                saved_count = 0
                
                for msg_data in messages:
                    # Check if message already exists
                    msg_id = uuid.UUID(msg_data['id'])
                    stmt = select(ChatMessage).where(ChatMessage.id == msg_id)
                    existing_msg = await session.scalar(stmt)
                    
                    if existing_msg:
                        continue  # Skip existing messages
                    
                    # Create new message
                    chat_message = ChatMessage(
                        id=msg_id,
                        session_id=session_uuid,
                        role=msg_data['role'],
                        content=msg_data['content'],
                        doc_meta=msg_data.get('metadata', {}),
                        parent_message_id=uuid.UUID(msg_data['parent_message_id']) if msg_data.get('parent_message_id') else None,
                        thread_id=uuid.UUID(msg_data.get('thread_id', str(uuid.uuid4())))
                    )
                    
                    session.add(chat_message)
                    saved_count += 1
                
                # Update message count in session
                if saved_count > 0:
                    stmt = select(ChatSession).where(ChatSession.id == session_uuid)
                    chat_session = await session.scalar(stmt)
                    if chat_session:
                        chat_session.message_count = chat_session.message_count + saved_count
                
                await session.commit()
                logger.info(f"Saved {saved_count} messages for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save chat messages for session {session_id}: {e}")
                await session.rollback()
                return False
    
    async def load_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load chat session from PostgreSQL.
        
        Args:
            session_id: UUID string of the session
            
        Returns:
            Session dictionary or None if not found
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(ChatSession).options(
                    selectinload(ChatSession.messages)
                ).where(ChatSession.id == uuid.UUID(session_id))
                
                chat_session = await session.scalar(stmt)
                
                if not chat_session:
                    return None
                
                # Convert to dictionary with messages
                session_dict = chat_session.to_dict()
                session_dict['messages'] = [
                    msg.to_dict() for msg in sorted(chat_session.messages, key=lambda m: m.created_at)
                ]
                
                return session_dict
                
            except Exception as e:
                logger.error(f"Failed to load chat session {session_id}: {e}")
                return None
    
    async def get_session_threads(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all threads in a chat session.
        
        Args:
            session_id: UUID string of the session
            
        Returns:
            List of thread summaries
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                # Get unique threads with basic stats
                stmt = text("""
                    SELECT 
                        thread_id,
                        COUNT(*) as message_count,
                        MIN(created_at) as created_at,
                        MAX(created_at) as last_message_at,
                        STRING_AGG(
                            CASE WHEN role = 'user' THEN LEFT(content, 50) ELSE NULL END, 
                            ' | ' ORDER BY created_at
                        ) as preview
                    FROM chat_messages 
                    WHERE session_id = :session_id 
                    GROUP BY thread_id
                    ORDER BY created_at DESC
                """)
                
                result = await session.execute(stmt, {"session_id": session_id})
                threads = []
                
                for row in result:
                    threads.append({
                        'thread_id': str(row.thread_id),
                        'message_count': row.message_count,
                        'created_at': row.created_at.isoformat() if row.created_at else None,
                        'last_message_at': row.last_message_at.isoformat() if row.last_message_at else None,
                        'preview': row.preview or ''
                    })
                
                return threads
                
            except Exception as e:
                logger.error(f"Failed to get session threads for {session_id}: {e}")
                return []
    
    async def get_thread_messages(
        self, 
        session_id: str, 
        thread_id: str,
        include_children: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all messages in a specific thread.
        
        Args:
            session_id: UUID string of the session
            thread_id: UUID string of the thread
            include_children: Whether to include child message relationships
            
        Returns:
            List of messages in thread order
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(ChatMessage).where(
                    and_(
                        ChatMessage.session_id == uuid.UUID(session_id),
                        ChatMessage.thread_id == uuid.UUID(thread_id)
                    )
                ).order_by(ChatMessage.created_at)
                
                result = await session.execute(stmt)
                messages = result.scalars().all()
                
                return [msg.to_dict(include_children=include_children) for msg in messages]
                
            except Exception as e:
                logger.error(f"Failed to get thread messages for {session_id}/{thread_id}: {e}")
                return []
    
    async def create_message_thread(
        self, 
        session_id: str, 
        parent_message_id: str,
        new_thread_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new thread branching from a parent message.
        
        Args:
            session_id: UUID string of the session
            parent_message_id: UUID string of the parent message
            new_thread_id: Optional UUID string for the new thread
            
        Returns:
            New thread ID or None if failed
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                # Verify parent message exists
                stmt = select(ChatMessage).where(
                    and_(
                        ChatMessage.id == uuid.UUID(parent_message_id),
                        ChatMessage.session_id == uuid.UUID(session_id)
                    )
                )
                parent_msg = await session.scalar(stmt)
                
                if not parent_msg:
                    logger.error(f"Parent message {parent_message_id} not found")
                    return None
                
                thread_id = new_thread_id or str(uuid.uuid4())
                logger.info(f"Created new thread {thread_id} from message {parent_message_id}")
                return thread_id
                
            except Exception as e:
                logger.error(f"Failed to create message thread: {e}")
                return None
    
    async def get_user_sessions(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get chat sessions for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            offset: Pagination offset
            
        Returns:
            List of session summaries
        """
        async with self.db_manager.get_postgres_session() as session:
            try:
                stmt = select(ChatSession).where(
                    ChatSession.user_id == user_id
                ).order_by(ChatSession.updated_at.desc()).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                sessions = result.scalars().all()
                
                return [s.to_dict() for s in sessions]
                
            except Exception as e:
                logger.error(f"Failed to get user sessions for {user_id}: {e}")
                return []


# Global instances
postgres_storage = PostgreSQLStorageService()
chat_storage = ChatStorageService()