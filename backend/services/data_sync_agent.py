"""
Automated Data Sync Agent
Runs on startup to populate database and continuously syncs data
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
import uuid

from config import settings
from services.fastf1_service import fastf1_service
from services.supabase_client import get_supabase_client


class DataSyncAgent:
    """
    Autonomous agent for syncing F1 data
    Runs automatically on application startup
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.scheduler = BackgroundScheduler()
        self.is_initialized = False
    
    async def initialize_database(self):
        """
        Initialize database with historical data if empty
        This runs automatically on startup
        """
        logger.info("🔍 Checking database status...")
        
        # Check if database has data
        drivers_count = await self._get_table_count('drivers')
        races_count = await self._get_table_count('races')
        
        if drivers_count == 0 or races_count == 0:
            logger.info("📊 Database is empty. Starting initial data load...")
            await self._populate_historical_data()
        else:
            logger.info(f"✅ Database already has data ({drivers_count} drivers, {races_count} races)")
            # Check for updates
            await self._check_for_updates()
        
        self.is_initialized = True
    
    async def _get_table_count(self, table_name: str) -> int:
        """Get row count for a table"""
        try:
            response = self.supabase.table(table_name).select('id', count='exact').execute()
            return response.count if hasattr(response, 'count') else 0
        except Exception as e:
            logger.error(f"Error counting {table_name}: {e}")
            return 0
    
    async def _populate_historical_data(self):
        """Populate database with historical F1 data"""
        logger.info("🚀 Starting historical data population...")
        
        for year in settings.HISTORICAL_YEARS:
            logger.info(f"📅 Processing {year} season...")
            
            try:
                # Insert season
                await self._insert_season(year)
                
                # Get season schedule
                schedule = fastf1_service.get_season_schedule(year)
                
                # Process each race
                for _, event in schedule.iterrows():
                    # Skip future races
                    if event['EventDate'] > datetime.now():
                        continue
                    
                    logger.info(f"  🏁 Processing {event['EventName']}...")
                    
                    try:
                        # Insert circuit
                        circuit_id = await self._insert_circuit(event)
                        
                        # Insert race
                        race_id = await self._insert_race(event, year, circuit_id)
                        
                        # Get and insert race results
                        await self._insert_race_results(year, event['RoundNumber'], race_id)
                        
                        logger.info(f"  ✅ {event['EventName']} complete")
                        
                    except Exception as e:
                        logger.error(f"  ❌ Error processing {event['EventName']}: {e}")
                        continue
                
                logger.info(f"✅ {year} season complete")
                
            except Exception as e:
                logger.error(f"❌ Error processing {year} season: {e}")
                continue
        
        # Populate drivers and constructors
        await self._populate_drivers_and_constructors()
        
        logger.info("🎉 Historical data population complete!")
    
    async def _insert_season(self, year: int):
        """Insert season record"""
        try:
            data = {
                'year': year,
                'url': f'https://ergast.com/api/f1/{year}'
            }
            self.supabase.table('seasons').upsert(data).execute()
        except Exception as e:
            logger.error(f"Error inserting season {year}: {e}")
    
    async def _insert_circuit(self, event) -> str:
        """Insert circuit and return ID"""
        try:
            circuit_data = {
                'id': str(uuid.uuid4()),
                'name': event['Location'],
                'location': event['Location'],
                'country': event['Country']
            }
            
            # Check if exists
            existing = self.supabase.table('circuits')\
                .select('id')\
                .eq('name', circuit_data['name'])\
                .execute()
            
            if existing.data:
                return existing.data[0]['id']
            
            result = self.supabase.table('circuits').insert(circuit_data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error inserting circuit: {e}")
            raise
    
    async def _insert_race(self, event, year: int, circuit_id: str) -> str:
        """Insert race and return ID"""
        try:
            race_data = {
                'id': str(uuid.uuid4()),
                'season_year': year,
                'round': int(event['RoundNumber']),
                'name': event['EventName'],
                'date': event['EventDate'].date().isoformat(),
                'time': event.get('Session5Date', event['EventDate']).time().isoformat() if hasattr(event.get('Session5Date', event['EventDate']), 'time') else '14:00:00',
                'circuit_id': circuit_id
            }
            
            # Check if exists
            existing = self.supabase.table('races')\
                .select('id')\
                .eq('season_year', year)\
                .eq('round', race_data['round'])\
                .execute()
            
            if existing.data:
                return existing.data[0]['id']
            
            result = self.supabase.table('races').insert(race_data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error inserting race: {e}")
            raise
    
    async def _insert_race_results(self, year: int, round_number: int, race_id: str):
        """Insert race results"""
        try:
            # Get race results from FastF1
            results = fastf1_service.get_race_results(year, round_number)
            
            for _, result in results.iterrows():
                # Get or create driver
                driver_id = await self._get_or_create_driver(result)
                
                # Get or create constructor
                constructor_id = await self._get_or_create_constructor(result)
                
                # Insert result
                result_data = {
                    'race_id': race_id,
                    'driver_id': driver_id,
                    'constructor_id': constructor_id,
                    'position': int(result['Position']) if pd.notna(result['Position']) else None,
                    'points': float(result['Points']) if pd.notna(result['Points']) else 0,
                    'status': result.get('Status', 'Finished'),
                    'fastest_lap_time': None  # Can be added later
                }
                
                self.supabase.table('race_results').insert(result_data).execute()
                
        except Exception as e:
            logger.error(f"Error inserting race results: {e}")
    
    async def _get_or_create_driver(self, result) -> str:
        """Get or create driver and return ID"""
        try:
            driver_code = result['Abbreviation']
            
            # Check if exists
            existing = self.supabase.table('drivers')\
                .select('id')\
                .eq('code', driver_code)\
                .execute()
            
            if existing.data:
                return existing.data[0]['id']
            
            # Create new driver
            driver_data = {
                'id': str(uuid.uuid4()),
                'name': result['FullName'],
                'code': driver_code,
                'permanent_number': int(result['DriverNumber']),
                'nationality': result.get('CountryCode', 'Unknown')
            }
            
            result = self.supabase.table('drivers').insert(driver_data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error getting/creating driver: {e}")
            raise
    
    async def _get_or_create_constructor(self, result) -> str:
        """Get or create constructor and return ID"""
        try:
            team_name = result['TeamName']
            
            # Check if exists
            existing = self.supabase.table('constructors')\
                .select('id')\
                .eq('name', team_name)\
                .execute()
            
            if existing.data:
                return existing.data[0]['id']
            
            # Create new constructor
            constructor_data = {
                'id': str(uuid.uuid4()),
                'name': team_name,
                'nationality': 'Unknown'
            }
            
            result = self.supabase.table('constructors').insert(constructor_data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error getting/creating constructor: {e}")
            raise
    
    async def _populate_drivers_and_constructors(self):
        """Populate current drivers and constructors"""
        try:
            current_year = datetime.now().year
            
            # Get drivers
            drivers = fastf1_service.get_current_drivers(current_year)
            for driver in drivers:
                await self._get_or_create_driver_from_dict(driver)
            
            # Get constructors
            constructors = fastf1_service.get_current_constructors(current_year)
            for constructor in constructors:
                await self._get_or_create_constructor_from_dict(constructor)
                
        except Exception as e:
            logger.error(f"Error populating drivers/constructors: {e}")
    
    async def _get_or_create_driver_from_dict(self, driver: Dict) -> str:
        """Get or create driver from dictionary"""
        try:
            existing = self.supabase.table('drivers')\
                .select('id')\
                .eq('code', driver['code'])\
                .execute()
            
            if existing.data:
                return existing.data[0]['id']
            
            driver_data = {
                'id': str(uuid.uuid4()),
                'name': driver['full_name'],
                'code': driver['code'],
                'permanent_number': driver['number'],
                'nationality': 'Unknown'
            }
            
            result = self.supabase.table('drivers').insert(driver_data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error creating driver: {e}")
            raise
    
    async def _get_or_create_constructor_from_dict(self, constructor: Dict) -> str:
        """Get or create constructor from dictionary"""
        try:
            existing = self.supabase.table('constructors')\
                .select('id')\
                .eq('name', constructor['name'])\
                .execute()
            
            if existing.data:
                return existing.data[0]['id']
            
            constructor_data = {
                'id': str(uuid.uuid4()),
                'name': constructor['name'],
                'nationality': 'Unknown'
            }
            
            result = self.supabase.table('constructors').insert(constructor_data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error creating constructor: {e}")
            raise
    
    async def _check_for_updates(self):
        """Check for new races or updates"""
        logger.info("🔍 Checking for updates...")
        
        try:
            current_year = datetime.now().year
            
            # Get latest race in database
            latest_race = self.supabase.table('races')\
                .select('date')\
                .order('date', desc=True)\
                .limit(1)\
                .execute()
            
            if not latest_race.data:
                return
            
            latest_date = datetime.fromisoformat(latest_race.data[0]['date'])
            
            # Check for new races
            schedule = fastf1_service.get_season_schedule(current_year)
            
            for _, event in schedule.iterrows():
                event_date = event['EventDate']
                if latest_date < event_date < datetime.now():
                    logger.info(f"📅 Found new race: {event['EventName']}")
                    # Process new race
                    # (implementation similar to _populate_historical_data)
            
            logger.info("✅ Update check complete")
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
    
    def start_background_sync(self):
        """Start background scheduler for periodic syncs"""
        logger.info("⏰ Starting background sync scheduler...")
        
        # Schedule daily sync
        self.scheduler.add_job(
            self._check_for_updates,
            'interval',
            hours=settings.SYNC_INTERVAL_HOURS,
            id='data_sync'
        )
        
        self.scheduler.start()
        logger.info(f"✅ Scheduler started (sync every {settings.SYNC_INTERVAL_HOURS} hours)")


import pandas as pd