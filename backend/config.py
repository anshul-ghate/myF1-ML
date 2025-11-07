"""
Configuration management for F1 Analytics Backend
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "F1 Analytics API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Supabase Settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_SERVICE_KEY: str
    
    # FastF1 Settings
    FASTF1_CACHE_DIR: str = "./fastf1_cache"
    FASTF1_ENABLE_CACHE: bool = True
    
    # ML Model Settings
    MODEL_DIR: str = "./models"
    MODEL_CACHE_DIR: str = "./models/cache"
    
    # Data Sync Settings
    AUTO_SYNC_ON_STARTUP: bool = True
    SYNC_INTERVAL_HOURS: int = 24
    HISTORICAL_YEARS: list = [2023, 2024]
    
    # CORS Settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://*.mgx.dev"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()