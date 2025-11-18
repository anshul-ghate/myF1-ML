"""
Supabase client singleton
"""
import sys
from supabase import create_client, Client
from config import settings
from loguru import logger


_supabase_client: Client = None


def get_supabase_client() -> Client:
    """Get or create Supabase client singleton"""
    global _supabase_client
    
    if _supabase_client is None:
        logger.info("Initializing Supabase client...")

        # Validate that the required environment variables are set
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            logger.error(
                "Missing Supabase credentials. Please set SUPABASE_URL and "
                "SUPABASE_SERVICE_KEY in your .env file."
            )
            sys.exit(1)

        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
        logger.info("✅ Supabase client initialized")
    
    return _supabase_client
