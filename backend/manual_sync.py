"""
Manual Data Sync Script
Run this script to manually populate or update the database with F1 data.
"""
import asyncio
from loguru import logger

from services.supabase_client import get_supabase_client
from services.data_sync_agent import DataSyncAgent

async def main():
    """
    Main function to run the data sync process.
    """
    logger.info("🚀 Starting manual data sync script...")

    try:
        # 1. Initialize Supabase client
        supabase_client = get_supabase_client()
        if not supabase_client:
            logger.error("❌ Failed to initialize Supabase client. Please check your .env file.")
            return

        # 2. Initialize the Data Sync Agent
        sync_agent = DataSyncAgent(supabase_client)

        # 3. Run the database initialization process
        # This will check if the database is empty and populate it, or check for updates.
        await sync_agent.initialize_database()

        logger.info("✅ Manual data sync finished successfully!")

    except Exception as e:
        logger.error(f"❌ An error occurred during the manual sync process: {e}")

if __name__ == "__main__":
    # Configure logging to write to a file
    logger.add("logs/manual_sync.log", rotation="10 MB", level="INFO")

    # Run the asynchronous main function
    asyncio.run(main())
