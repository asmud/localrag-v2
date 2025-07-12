from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import chat, documents, health, rag, jobs
from core.config import settings
from core.database import db_manager
from core.logging import setup_logging
from core.models import model_manager
from core.llm_service import llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup logging first
    setup_logging()
    logger.info("Starting LocalRAG application...")
    
    # Initialize background workers manager
    worker_manager = None
    
    try:
        await db_manager.initialize()
        logger.info("Database connections initialized")
        
        await model_manager.initialize()
        logger.info("Models initialized")
        
        await llm_service.initialize()
        logger.info("LLM service initialized")
        
        # Initialize background workers if enabled
        if settings.task_queue_enabled and settings.task_workers_auto_start:
            try:
                from core.background_workers import worker_manager as wm
                from core.job_manager import job_manager
                
                worker_manager = wm
                
                # Ensure job tracking tables exist
                await job_manager.create_job_tables()
                logger.info("Job tracking tables created/verified")
                
                # Start background workers
                await worker_manager.start_workers()
                logger.info("Background workers started successfully")
                
            except Exception as e:
                logger.warning(f"Failed to start background workers: {e}")
                logger.warning("API will continue running without background processing")
                worker_manager = None
        else:
            logger.info("Background workers disabled in configuration")
        
        logger.info("LocalRAG application started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        logger.info("Shutting down LocalRAG application...")
        
        # Stop background workers gracefully
        if worker_manager and worker_manager.is_running:
            try:
                logger.info("Stopping background workers...")
                await worker_manager.stop_workers()
                logger.info("Background workers stopped")
            except Exception as e:
                logger.error(f"Error stopping background workers: {e}")
        
        await model_manager.cleanup()
        await db_manager.close()
        
        logger.info("LocalRAG application shut down complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade local RAG system with knowledge graph integration",
    lifespan=lifespan,
    debug=settings.debug,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix=settings.api_v1_prefix, tags=["health"])
app.include_router(documents.router, prefix=settings.api_v1_prefix, tags=["documents"])
app.include_router(jobs.router, prefix=settings.api_v1_prefix, tags=["jobs"])
app.include_router(rag.router, prefix=settings.api_v1_prefix, tags=["rag"])
app.include_router(chat.router, prefix=settings.api_v1_prefix, tags=["chat"])


@app.get("/")
async def root():
    return {
        "message": "LocalRAG API",
        "version": settings.app_version,
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )