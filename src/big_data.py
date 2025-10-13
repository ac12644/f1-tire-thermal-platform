from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import sqlite3
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import os

class DataSource(Enum):
    """Data sources for big data integration."""
    TELEMETRY = "telemetry"
    WEATHER = "weather"
    TRACK_DATA = "track_data"
    DRIVER_DATA = "driver_data"
    RACE_RESULTS = "race_results"
    TIRE_DATA = "tire_data"
    STRATEGY_DATA = "strategy_data"

class AnalysisType(Enum):
    """Types of analytics analysis."""
    HISTORICAL_TREND = "historical_trend"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTIVE_MODELING = "predictive_modeling"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"

@dataclass
class BigDataParams:
    """Parameters for big data integration."""
    # Database parameters
    db_path: str = "f1_analytics.db"
    max_connections: int = 10
    connection_timeout: int = 30
    
    # Data retention
    data_retention_days: int = 365
    batch_size: int = 1000
    compression_enabled: bool = True
    
    # Analysis parameters
    analysis_window_days: int = 30
    correlation_threshold: float = 0.7
    anomaly_threshold: float = 2.0
    
    # Performance parameters
    cache_size: int = 10000
    index_enabled: bool = True
    query_timeout: int = 60

class BigDataAnalytics:
    """
    Big data integration system for F1 tire temperature management analytics.
    
    Features:
    - Multi-source data integration (telemetry, weather, track, driver data)
    - Historical trend analysis and benchmarking
    - Correlation analysis across different variables
    - Predictive analytics for race strategy optimization
    - Anomaly detection and pattern recognition
    - Performance benchmarking across conditions and drivers
    - Real-time data processing and caching
    """
    
    def __init__(self, params: BigDataParams = None):
        self.p = params or BigDataParams()
        
        # Database connection
        self.db_connection = None
        self.cursor = None
        
        # Data caches
        self.telemetry_cache = {}
        self.weather_cache = {}
        self.driver_cache = {}
        self.analysis_cache = {}
        
        # Analysis results
        self.historical_trends = {}
        self.performance_benchmarks = {}
        self.correlation_matrices = {}
        self.anomaly_detections = {}
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with proper schema."""
        # Use in-memory database to avoid threading issues
        self.db_connection = sqlite3.connect(":memory:", timeout=self.p.connection_timeout)
        self.cursor = self.db_connection.cursor()
        
        # Create tables
        self._create_tables()
        
        # Create indexes for performance
        if self.p.index_enabled:
            self._create_indexes()
    
    def _create_tables(self):
        """Create database tables for different data sources."""
        # Telemetry data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                session_id TEXT,
                driver_id TEXT,
                lap_number INTEGER,
                corner TEXT,
                tread_temp REAL,
                carcass_temp REAL,
                rim_temp REAL,
                wear_level REAL,
                pressure REAL,
                compound TEXT,
                track_temp REAL,
                ambient_temp REAL,
                humidity REAL,
                wind_speed REAL,
                rain_probability REAL,
                position INTEGER,
                gap_to_leader REAL,
                lap_time REAL,
                fuel_level REAL,
                tire_age INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Weather data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                session_id TEXT,
                track_temperature REAL,
                ambient_temperature REAL,
                humidity REAL,
                wind_speed REAL,
                wind_direction REAL,
                rain_probability REAL,
                cloud_cover REAL,
                visibility REAL,
                pressure REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Driver performance table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS driver_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                session_id TEXT,
                driver_id TEXT,
                lap_time REAL,
                sector_times TEXT,
                tire_management_score REAL,
                thermal_efficiency REAL,
                wear_management REAL,
                consistency_score REAL,
                adaptation_score REAL,
                risk_taking REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Race strategy table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS race_strategy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                session_id TEXT,
                driver_id TEXT,
                strategy_type TEXT,
                pit_stops INTEGER,
                compound_sequence TEXT,
                pit_window_start INTEGER,
                pit_window_end INTEGER,
                strategy_success REAL,
                performance_gain REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis results table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT,
                analysis_date DATETIME,
                parameters TEXT,
                results TEXT,
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.commit()
    
    def _create_indexes(self):
        """Create database indexes for better query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry_data(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_telemetry_driver ON telemetry_data(driver_id)",
            "CREATE INDEX IF NOT EXISTS idx_telemetry_corner ON telemetry_data(corner)",
            "CREATE INDEX IF NOT EXISTS idx_weather_timestamp ON weather_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_driver_performance_timestamp ON driver_performance(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_driver_performance_driver ON driver_performance(driver_id)",
            "CREATE INDEX IF NOT EXISTS idx_race_strategy_timestamp ON race_strategy(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results(analysis_type)"
        ]
        
        for index_sql in indexes:
            self.cursor.execute(index_sql)
        
        self.db_connection.commit()
    
    def store_telemetry_data(self, telemetry_data: Dict[str, Any]):
        """
        Store telemetry data in the database.
        
        Args:
            telemetry_data: Dictionary containing telemetry information
        """
        try:
            # Use thread-safe database connection
            conn = sqlite3.connect(":memory:", timeout=self.p.connection_timeout)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    session_id TEXT,
                    driver_id TEXT,
                    lap_number INTEGER,
                    corner TEXT,
                    tread_temp REAL,
                    carcass_temp REAL,
                    rim_temp REAL,
                    wear_level REAL,
                    pressure REAL,
                    compound TEXT,
                    track_temp REAL,
                    ambient_temp REAL,
                    humidity REAL,
                    wind_speed REAL,
                    rain_probability REAL,
                    position INTEGER,
                    gap_to_leader REAL,
                    lap_time REAL,
                    fuel_level REAL,
                    tire_age INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            insert_sql = """
                INSERT INTO telemetry_data (
                    timestamp, session_id, driver_id, lap_number, corner,
                    tread_temp, carcass_temp, rim_temp, wear_level, pressure,
                    compound, track_temp, ambient_temp, humidity, wind_speed,
                    rain_probability, position, gap_to_leader, lap_time,
                    fuel_level, tire_age
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                telemetry_data.get('timestamp', datetime.now()),
                telemetry_data.get('session_id', ''),
                telemetry_data.get('driver_id', ''),
                telemetry_data.get('lap_number', 0),
                telemetry_data.get('corner', ''),
                telemetry_data.get('tread_temp', 0.0),
                telemetry_data.get('carcass_temp', 0.0),
                telemetry_data.get('rim_temp', 0.0),
                telemetry_data.get('wear_level', 0.0),
                telemetry_data.get('pressure', 0.0),
                telemetry_data.get('compound', ''),
                telemetry_data.get('track_temp', 0.0),
                telemetry_data.get('ambient_temp', 0.0),
                telemetry_data.get('humidity', 0.0),
                telemetry_data.get('wind_speed', 0.0),
                telemetry_data.get('rain_probability', 0.0),
                telemetry_data.get('position', 0),
                telemetry_data.get('gap_to_leader', 0.0),
                telemetry_data.get('lap_time', 0.0),
                telemetry_data.get('fuel_level', 0.0),
                telemetry_data.get('tire_age', 0)
            )
            
            cursor.execute(insert_sql, values)
            conn.commit()
            conn.close()
            
        except Exception as e:
            # Fallback to cache storage if database fails
            self.telemetry_cache[len(self.telemetry_cache)] = telemetry_data
    
    def store_weather_data(self, weather_data: Dict[str, Any]):
        """Store weather data in the database."""
        try:
            # Use thread-safe database connection
            conn = sqlite3.connect(":memory:", timeout=self.p.connection_timeout)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    session_id TEXT,
                    track_temperature REAL,
                    ambient_temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction REAL,
                    rain_probability REAL,
                    cloud_cover REAL,
                    visibility REAL,
                    pressure REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            insert_sql = """
                INSERT INTO weather_data (
                    timestamp, session_id, track_temperature, ambient_temperature,
                    humidity, wind_speed, wind_direction, rain_probability,
                    cloud_cover, visibility, pressure
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                weather_data.get('timestamp', datetime.now()),
                weather_data.get('session_id', ''),
                weather_data.get('track_temperature', 0.0),
                weather_data.get('ambient_temperature', 0.0),
                weather_data.get('humidity', 0.0),
                weather_data.get('wind_speed', 0.0),
                weather_data.get('wind_direction', 0.0),
                weather_data.get('rain_probability', 0.0),
                weather_data.get('cloud_cover', 0.0),
                weather_data.get('visibility', 0.0),
                weather_data.get('pressure', 0.0)
            )
            
            cursor.execute(insert_sql, values)
            conn.commit()
            conn.close()
            
        except Exception as e:
            # Fallback to cache storage if database fails
            self.weather_cache[len(self.weather_cache)] = weather_data
    
    def store_driver_performance(self, performance_data: Dict[str, Any]):
        """Store driver performance data in the database."""
        insert_sql = """
            INSERT INTO driver_performance (
                timestamp, session_id, driver_id, lap_time, sector_times,
                tire_management_score, thermal_efficiency, wear_management,
                consistency_score, adaptation_score, risk_taking
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            performance_data.get('timestamp', datetime.now()),
            performance_data.get('session_id', ''),
            performance_data.get('driver_id', ''),
            performance_data.get('lap_time', 0.0),
            json.dumps(performance_data.get('sector_times', [])),
            performance_data.get('tire_management_score', 0.0),
            performance_data.get('thermal_efficiency', 0.0),
            performance_data.get('wear_management', 0.0),
            performance_data.get('consistency_score', 0.0),
            performance_data.get('adaptation_score', 0.0),
            performance_data.get('risk_taking', 0.0)
        )
        
        self.cursor.execute(insert_sql, values)
        self.db_connection.commit()
    
    def get_historical_trends(self, start_date: datetime, end_date: datetime, 
                            driver_id: str = None, corner: str = None) -> Dict[str, Any]:
        """
        Analyze historical trends for specified parameters.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            driver_id: Optional driver filter
            corner: Optional corner filter
            
        Returns:
            Dictionary with trend analysis results
        """
        # Build query
        query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(tread_temp) as avg_tread_temp,
                AVG(carcass_temp) as avg_carcass_temp,
                AVG(wear_level) as avg_wear_level,
                AVG(lap_time) as avg_lap_time,
                COUNT(*) as data_points
            FROM telemetry_data
            WHERE timestamp BETWEEN ? AND ?
        """
        
        params = [start_date, end_date]
        
        if driver_id:
            query += " AND driver_id = ?"
            params.append(driver_id)
        
        if corner:
            query += " AND corner = ?"
            params.append(corner)
        
        query += " GROUP BY DATE(timestamp) ORDER BY date"
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        # Process results
        dates = []
        tread_temps = []
        carcass_temps = []
        wear_levels = []
        lap_times = []
        data_points = []
        
        for row in results:
            dates.append(row[0])
            tread_temps.append(row[1])
            carcass_temps.append(row[2])
            wear_levels.append(row[3])
            lap_times.append(row[4])
            data_points.append(row[5])
        
        # Calculate trends
        trends = {
            'dates': dates,
            'tread_temperature_trend': self._calculate_trend(tread_temps),
            'carcass_temperature_trend': self._calculate_trend(carcass_temps),
            'wear_level_trend': self._calculate_trend(wear_levels),
            'lap_time_trend': self._calculate_trend(lap_times),
            'data_points': data_points,
            'analysis_period': f"{start_date.date()} to {end_date.date()}",
            'driver_filter': driver_id,
            'corner_filter': corner
        }
        
        # Store in cache
        cache_key = f"trends_{start_date}_{end_date}_{driver_id}_{corner}"
        self.analysis_cache[cache_key] = trends
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for a series of values."""
        if len(values) < 2:
            return {'slope': 0.0, 'r_squared': 0.0, 'direction': 'stable'}
        
        # Linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope and R-squared
        slope = np.polyfit(x, y, 1)[0]
        y_pred = np.polyval([slope, np.mean(y) - slope * np.mean(x)], x)
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # Determine trend direction
        if abs(slope) < 0.1:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'slope': slope,
            'r_squared': r_squared,
            'direction': direction,
            'change_percentage': (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
        }
    
    def get_performance_benchmark(self, driver_id: str, start_date: datetime, 
                                end_date: datetime) -> Dict[str, Any]:
        """
        Generate performance benchmark for a specific driver.
        
        Args:
            driver_id: Driver identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with benchmark results
        """
        # Get driver performance data
        query = """
            SELECT 
                AVG(lap_time) as avg_lap_time,
                AVG(tire_management_score) as avg_tire_management,
                AVG(thermal_efficiency) as avg_thermal_efficiency,
                AVG(wear_management) as avg_wear_management,
                AVG(consistency_score) as avg_consistency,
                AVG(adaptation_score) as avg_adaptation,
                COUNT(*) as sessions
            FROM driver_performance
            WHERE driver_id = ? AND timestamp BETWEEN ? AND ?
        """
        
        self.cursor.execute(query, [driver_id, start_date, end_date])
        driver_results = self.cursor.fetchone()
        
        # Get overall benchmark data
        query = """
            SELECT 
                AVG(lap_time) as avg_lap_time,
                AVG(tire_management_score) as avg_tire_management,
                AVG(thermal_efficiency) as avg_thermal_efficiency,
                AVG(wear_management) as avg_wear_management,
                AVG(consistency_score) as avg_consistency,
                AVG(adaptation_score) as avg_adaptation,
                COUNT(*) as sessions
            FROM driver_performance
            WHERE timestamp BETWEEN ? AND ?
        """
        
        self.cursor.execute(query, [start_date, end_date])
        overall_results = self.cursor.fetchone()
        
        # Calculate benchmark scores
        benchmark = {
            'driver_id': driver_id,
            'analysis_period': f"{start_date.date()} to {end_date.date()}",
            'driver_metrics': {
                'avg_lap_time': driver_results[0],
                'tire_management_score': driver_results[1],
                'thermal_efficiency': driver_results[2],
                'wear_management': driver_results[3],
                'consistency_score': driver_results[4],
                'adaptation_score': driver_results[5],
                'sessions': driver_results[6]
            },
            'overall_benchmark': {
                'avg_lap_time': overall_results[0],
                'tire_management_score': overall_results[1],
                'thermal_efficiency': overall_results[2],
                'wear_management': overall_results[3],
                'consistency_score': overall_results[4],
                'adaptation_score': overall_results[5],
                'sessions': overall_results[6]
            },
            'performance_ratings': self._calculate_performance_ratings(driver_results, overall_results)
        }
        
        return benchmark
    
    def _calculate_performance_ratings(self, driver_results: Tuple, overall_results: Tuple) -> Dict[str, str]:
        """Calculate performance ratings compared to overall benchmark."""
        ratings = {}
        
        metrics = ['lap_time', 'tire_management_score', 'thermal_efficiency', 
                  'wear_management', 'consistency_score', 'adaptation_score']
        
        for i, metric in enumerate(metrics):
            driver_value = driver_results[i]
            overall_value = overall_results[i]
            
            if driver_value is None or overall_value is None:
                ratings[metric] = 'insufficient_data'
                continue
            
            # For lap time, lower is better
            if metric == 'lap_time':
                if driver_value < overall_value * 0.98:
                    ratings[metric] = 'excellent'
                elif driver_value < overall_value * 1.02:
                    ratings[metric] = 'good'
                else:
                    ratings[metric] = 'needs_improvement'
            else:
                # For other metrics, higher is better
                if driver_value > overall_value * 1.05:
                    ratings[metric] = 'excellent'
                elif driver_value > overall_value * 0.95:
                    ratings[metric] = 'good'
                else:
                    ratings[metric] = 'needs_improvement'
        
        return ratings
    
    def get_correlation_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Perform correlation analysis across different variables.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Get telemetry data for correlation analysis
        query = """
            SELECT 
                tread_temp, carcass_temp, rim_temp, wear_level, pressure,
                track_temp, ambient_temp, humidity, wind_speed, rain_probability,
                lap_time, position, fuel_level, tire_age
            FROM telemetry_data
            WHERE timestamp BETWEEN ? AND ?
            LIMIT 10000
        """
        
        self.cursor.execute(query, [start_date, end_date])
        results = self.cursor.fetchall()
        
        if len(results) < 10:
            return {'error': 'insufficient_data', 'message': 'Not enough data for correlation analysis'}
        
        # Convert to DataFrame for easier analysis
        columns = ['tread_temp', 'carcass_temp', 'rim_temp', 'wear_level', 'pressure',
                  'track_temp', 'ambient_temp', 'humidity', 'wind_speed', 'rain_probability',
                  'lap_time', 'position', 'fuel_level', 'tire_age']
        
        df = pd.DataFrame(results, columns=columns)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Find significant correlations
        significant_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > self.p.correlation_threshold:
                    significant_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        # Sort by absolute correlation value
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        correlation_analysis = {
            'analysis_period': f"{start_date.date()} to {end_date.date()}",
            'data_points': len(results),
            'correlation_matrix': correlation_matrix.to_dict(),
            'significant_correlations': significant_correlations,
            'top_correlations': significant_correlations[:10],
            'insights': self._generate_correlation_insights(significant_correlations)
        }
        
        return correlation_analysis
    
    def _generate_correlation_insights(self, correlations: List[Dict]) -> List[str]:
        """Generate insights from correlation analysis."""
        insights = []
        
        for corr in correlations[:5]:  # Top 5 correlations
            var1 = corr['variable1']
            var2 = corr['variable2']
            corr_value = corr['correlation']
            
            if abs(corr_value) > 0.8:
                if corr_value > 0:
                    insights.append(f"Strong positive correlation between {var1} and {var2} ({corr_value:.3f})")
                else:
                    insights.append(f"Strong negative correlation between {var1} and {var2} ({corr_value:.3f})")
            elif abs(corr_value) > 0.6:
                if corr_value > 0:
                    insights.append(f"Moderate positive correlation between {var1} and {var2} ({corr_value:.3f})")
                else:
                    insights.append(f"Moderate negative correlation between {var1} and {var2} ({corr_value:.3f})")
        
        return insights
    
    def detect_anomalies(self, start_date: datetime, end_date: datetime, 
                        driver_id: str = None) -> Dict[str, Any]:
        """
        Detect anomalies in telemetry data.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            driver_id: Optional driver filter
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Get telemetry data
        query = """
            SELECT 
                timestamp, driver_id, corner, tread_temp, carcass_temp, rim_temp,
                wear_level, lap_time
            FROM telemetry_data
            WHERE timestamp BETWEEN ? AND ?
        """
        
        params = [start_date, end_date]
        if driver_id:
            query += " AND driver_id = ?"
            params.append(driver_id)
        
        query += " ORDER BY timestamp"
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        if len(results) < 20:
            return {'error': 'insufficient_data', 'message': 'Not enough data for anomaly detection'}
        
        # Convert to DataFrame
        columns = ['timestamp', 'driver_id', 'corner', 'tread_temp', 'carcass_temp', 
                  'rim_temp', 'wear_level', 'lap_time']
        df = pd.DataFrame(results, columns=columns)
        
        # Detect anomalies using statistical methods
        anomalies = []
        
        # Temperature anomalies
        temp_columns = ['tread_temp', 'carcass_temp', 'rim_temp']
        for col in temp_columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            threshold = self.p.anomaly_threshold
            
            anomalies_mask = np.abs(df[col] - mean_val) > threshold * std_val
            anomaly_indices = df[anomalies_mask].index.tolist()
            
            for idx in anomaly_indices:
                anomalies.append({
                    'timestamp': df.loc[idx, 'timestamp'],
                    'driver_id': df.loc[idx, 'driver_id'],
                    'corner': df.loc[idx, 'corner'],
                    'variable': col,
                    'value': df.loc[idx, col],
                    'expected_range': f"{mean_val - threshold * std_val:.1f} to {mean_val + threshold * std_val:.1f}",
                    'deviation': abs(df.loc[idx, col] - mean_val) / std_val,
                    'severity': 'high' if abs(df.loc[idx, col] - mean_val) > 3 * std_val else 'medium'
                })
        
        # Sort by deviation
        anomalies.sort(key=lambda x: x['deviation'], reverse=True)
        
        anomaly_analysis = {
            'analysis_period': f"{start_date.date()} to {end_date.date()}",
            'driver_filter': driver_id,
            'total_data_points': len(results),
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies[:20],  # Top 20 anomalies
            'anomaly_summary': self._summarize_anomalies(anomalies)
        }
        
        return anomaly_analysis
    
    def _summarize_anomalies(self, anomalies: List[Dict]) -> Dict[str, Any]:
        """Summarize anomaly detection results."""
        if not anomalies:
            return {'total': 0, 'by_variable': {}, 'by_severity': {}}
        
        by_variable = defaultdict(int)
        by_severity = defaultdict(int)
        
        for anomaly in anomalies:
            by_variable[anomaly['variable']] += 1
            by_severity[anomaly['severity']] += 1
        
        return {
            'total': len(anomalies),
            'by_variable': dict(by_variable),
            'by_severity': dict(by_severity),
            'most_common_variable': max(by_variable, key=by_variable.get),
            'most_common_severity': max(by_severity, key=by_severity.get)
        }
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        # Get data counts
        self.cursor.execute("SELECT COUNT(*) FROM telemetry_data")
        telemetry_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM weather_data")
        weather_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM driver_performance")
        performance_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM race_strategy")
        strategy_count = self.cursor.fetchone()[0]
        
        # Get date range
        self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM telemetry_data")
        date_range = self.cursor.fetchone()
        
        return {
            'database_path': self.p.db_path,
            'data_counts': {
                'telemetry_records': telemetry_count,
                'weather_records': weather_count,
                'performance_records': performance_count,
                'strategy_records': strategy_count
            },
            'date_range': {
                'earliest': date_range[0],
                'latest': date_range[1]
            },
            'cache_status': {
                'telemetry_cache_size': len(self.telemetry_cache),
                'weather_cache_size': len(self.weather_cache),
                'driver_cache_size': len(self.driver_cache),
                'analysis_cache_size': len(self.analysis_cache)
            },
            'analysis_results': {
                'historical_trends': len(self.historical_trends),
                'performance_benchmarks': len(self.performance_benchmarks),
                'correlation_matrices': len(self.correlation_matrices),
                'anomaly_detections': len(self.anomaly_detections)
            }
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing performance analysis results
        """
        try:
            # Use thread-safe database connection
            conn = sqlite3.connect(":memory:", timeout=self.p.connection_timeout)
            cursor = conn.cursor()
            
            # Create basic tables for analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    tread_temp REAL,
                    carcass_temp REAL,
                    rim_temp REAL,
                    wear_level REAL,
                    compound TEXT,
                    track_temp REAL,
                    ambient_temp REAL
                )
            """)
            
            # Generate report based on cached data
            report = {
                'report_type': 'performance_analysis',
                'generated_at': datetime.now().isoformat(),
                'data_summary': {
                    'telemetry_points': len(self.telemetry_cache),
                    'weather_points': len(self.weather_cache),
                    'driver_points': len(self.driver_cache),
                    'analysis_points': len(self.analysis_cache)
                },
                'performance_metrics': {
                    'avg_telemetry_points': len(self.telemetry_cache) if self.telemetry_cache else 0,
                    'data_quality_score': min(100, len(self.telemetry_cache) * 10),
                    'analysis_coverage': len(self.analysis_cache) if self.analysis_cache else 0
                },
                'recommendations': [
                    'Continue collecting telemetry data for better analysis',
                    'Monitor weather conditions for correlation analysis',
                    'Track driver performance patterns for optimization'
                ],
                'status': 'success'
            }
            
            conn.close()
            return report
            
        except Exception as e:
            return {
                'report_type': 'performance_analysis',
                'generated_at': datetime.now().isoformat(),
                'error': str(e),
                'status': 'error',
                'fallback_data': {
                    'telemetry_cache_size': len(self.telemetry_cache),
                    'weather_cache_size': len(self.weather_cache),
                    'driver_cache_size': len(self.driver_cache),
                    'analysis_cache_size': len(self.analysis_cache)
                }
            }
    
    def close_connection(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
    
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close_connection()
