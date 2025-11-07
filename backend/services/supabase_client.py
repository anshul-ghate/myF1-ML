"""
Supabase client singleton
"""
from supabase import create_client, Client
from config import settings
from loguru import logger


_supabase_client: Client = None


def get_supabase_client() -> Client:
    """Get or create Supabase client singleton"""
    global _supabase_client
    
    if _supabase_client is None:
        logger.info("Initializing Supabase client...")
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
        logger.info("✅ Supabase client initialized")
    
    return _supabase_client