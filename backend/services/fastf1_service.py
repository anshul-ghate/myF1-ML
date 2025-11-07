"""
FastF1 Data Service
Handles all interactions with the FastF1 library
"""
import fastf1
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime
from loguru import logger
from config import settings
import os


class FastF1Service:
    """Service for fetching F1 data using FastF1 library"""
    
    def __init__(self):
        """Initialize FastF1 service with caching"""
        if settings.FASTF1_ENABLE_CACHE:
            # Create cache directory if it doesn't exist
            os.makedirs(settings.FASTF1_CACHE_DIR, exist_ok=True)
            fastf1.Cache.enable_cache(settings.FASTF1_CACHE_DIR)
            logger.info(f"FastF1 cache enabled at {settings.FASTF1_CACHE_DIR}")
    
    def get_season_schedule(self, year: int) -> pd.DataFrame:
        """
        Get complete season schedule
        
        Args:
            year: Season year
            
        Returns:
            DataFrame with race schedule
        """
        try:
            logger.info(f"Fetching {year} season schedule...")
            schedule = fastf1.get_event_schedule(year)
            logger.info(f"✅ Found {len(schedule)} races for {year}")
            return schedule
        except Exception as e:
            logger.error(f"Error fetching season schedule: {e}")
            raise
    
    def get_session(self, year: int, round_number: int, session_type: str = 'R'):
        """
        Get session data
        
        Args:
            year: Season year
            round_number: Race round number
            session_type: 'FP1', 'FP2', 'FP3', 'Q', 'S' (Sprint), 'R' (Race)
            
        Returns:
            FastF1 Session object
        """
        try:
            logger.info(f"Fetching {year} Round {round_number} {session_type}...")
            session = fastf1.get_session(year, round_number, session_type)
            session.load()
            logger.info(f"✅ Session loaded successfully")
            return session
        except Exception as e:
            logger.error(f"Error fetching session: {e}")
            raise
    
    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        """
        Get race results
        
        Args:
            year: Season year
            round_number: Race round number
            
        Returns:
            DataFrame with race results
        """
        try:
            session = self.get_session(year, round_number, 'R')
            results = session.results
            logger.info(f"✅ Got results for {len(results)} drivers")
            return results
        except Exception as e:
            logger.error(f"Error fetching race results: {e}")
            raise
    
    def get_driver_laps(self, session, driver_code: str) -> pd.DataFrame:
        """
        Get all laps for a specific driver
        
        Args:
            session: FastF1 Session object
            driver_code: Driver abbreviation (e.g., 'VER', 'HAM')
            
        Returns:
            DataFrame with driver laps
        """
        try:
            driver_laps = session.laps.pick_driver(driver_code)
            logger.info(f"✅ Got {len(driver_laps)} laps for {driver_code}")
            return driver_laps
        except Exception as e:
            logger.error(f"Error fetching driver laps: {e}")
            raise
    
    def get_lap_telemetry(self, lap) -> pd.DataFrame:
        """
        Get telemetry for a specific lap
        
        Args:
            lap: FastF1 Lap object
            
        Returns:
            DataFrame with telemetry data
        """
        try:
            telemetry = lap.get_car_data()
            telemetry = telemetry.add_distance()
            return telemetry
        except Exception as e:
            logger.error(f"Error fetching lap telemetry: {e}")
            raise
    
    def get_current_drivers(self, year: int) -> List[Dict]:
        """
        Get current season drivers
        
        Args:
            year: Season year
            
        Returns:
            List of driver dictionaries
        """
        try:
            # Get first race of the season
            session = self.get_session(year, 1, 'R')
            results = session.results
            
            drivers = []
            for _, driver in results.iterrows():
                drivers.append({
                    'code': driver['Abbreviation'],
                    'number': int(driver['DriverNumber']),
                    'full_name': driver['FullName'],
                    'team': driver['TeamName'],
                    'team_color': driver.get('TeamColor', '#FFFFFF')
                })
            
            logger.info(f"✅ Found {len(drivers)} drivers for {year}")
            return drivers
            
        except Exception as e:
            logger.error(f"Error fetching current drivers: {e}")
            raise
    
    def get_current_constructors(self, year: int) -> List[Dict]:
        """
        Get current season constructors
        
        Args:
            year: Season year
            
        Returns:
            List of constructor dictionaries
        """
        try:
            session = self.get_session(year, 1, 'R')
            results = session.results
            
            # Get unique teams
            teams = results['TeamName'].unique()
            
            constructors = []
            for team in teams:
                team_data = results[results['TeamName'] == team].iloc[0]
                constructors.append({
                    'name': team,
                    'color': team_data.get('TeamColor', '#FFFFFF')
                })
            
            logger.info(f"✅ Found {len(constructors)} constructors for {year}")
            return constructors
            
        except Exception as e:
            logger.error(f"Error fetching current constructors: {e}")
            raise


# Global service instance
fastf1_service = FastF1Service()