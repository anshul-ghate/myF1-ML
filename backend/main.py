"""
F1 Analytics FastAPI Backend
Main application entry point with automated startup agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from config import settings
from services.data_sync_agent import DataSyncAgent
from services.supabase_client import get_supabase_client
from api import (
    races,
    predictions,
    strategy,
    drivers,
    constructors,
    analytics
)


# Automated startup agent
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - runs on startup and shutdown
    This is where the automated agent runs to populate the database
    """
    logger.info("🚀 Starting F1 Analytics Backend...")
    
    # Initialize services
    supabase = get_supabase_client()
    
    if settings.AUTO_SYNC_ON_STARTUP:
        logger.info("🤖 Starting automated data sync agent...")
        agent = DataSyncAgent(supabase)
        
        try:
            # Check if database needs initialization
            await agent.initialize_database()
            logger.info("✅ Database initialization complete")
            
            # Start background sync scheduler
            agent.start_background_sync()
            logger.info("✅ Background sync scheduler started")
            
        except Exception as e:
            logger.error(f"❌ Error during startup: {e}")
            # Don't fail startup, but log the error
    
    logger.info("✅ F1 Analytics Backend ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down F1 Analytics Backend...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "service": "F1 Analytics API"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "F1 Analytics API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(races.router, prefix=f"{settings.API_PREFIX}/races", tags=["Races"])
app.include_router(predictions.router, prefix=f"{settings.API_PREFIX}/predictions", tags=["Predictions"])
app.include_router(strategy.router, prefix=f"{settings.API_PREFIX}/strategy", tags=["Strategy"])
app.include_router(drivers.router, prefix=f"{settings.API_PREFIX}/drivers", tags=["Drivers"])
app.include_router(constructors.router, prefix=f"{settings.API_PREFIX}/constructors", tags=["Constructors"])
app.include_router(analytics.router, prefix=f"{settings.API_PREFIX}/analytics", tags=["Analytics"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )