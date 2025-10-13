from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
from collections import defaultdict
import statistics
import threading
import time
import uuid
import warnings
warnings.filterwarnings('ignore')

class ReportType(Enum):
    """Types of reports."""
    RACE_SUMMARY = "race_summary"
    TIRE_ANALYSIS = "tire_analysis"
    STRATEGY_REPORT = "strategy_report"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    WEATHER_IMPACT = "weather_impact"
    DRIVER_COMPARISON = "driver_comparison"
    PREDICTIVE_INSIGHTS = "predictive_insights"
    COLLABORATION_REPORT = "collaboration_report"
    TECHNICAL_REPORT = "technical_report"
    EXECUTIVE_SUMMARY = "executive_summary"

class ReportFormat(Enum):
    """Report formats."""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"
    POWERPOINT = "powerpoint"

class Priority(Enum):
    """Priority levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class InsightCategory(Enum):
    """Categories of insights."""
    PERFORMANCE = "performance"
    STRATEGY = "strategy"
    TECHNICAL = "technical"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"
    TREND = "trend"

@dataclass
class ReportParams:
    """Parameters for report generation."""
    # Report configuration
    auto_generate: bool = True
    generation_interval: int = 300  # seconds
    retention_days: int = 30
    
    # Content parameters
    include_charts: bool = True
    include_tables: bool = True
    include_predictions: bool = True
    include_recommendations: bool = True
    
    # Format parameters
    default_format: ReportFormat = ReportFormat.HTML
    chart_style: str = "seaborn"
    color_scheme: str = "F1"
    
    # Data parameters
    data_sources: List[str] = None
    time_range_hours: int = 24
    include_historical: bool = True
    
    # Quality parameters
    min_data_points: int = 10
    confidence_threshold: float = 0.7
    validation_required: bool = True

@dataclass
class Insight:
    """Represents an automated insight."""
    insight_id: str
    title: str
    description: str
    category: InsightCategory
    priority: Priority
    confidence: float
    impact_score: float
    data_points: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    tags: List[str] = None

class ReportGenerator:
    """
    Advanced reporting system with automated insights.
    
    Features:
    - Automated report generation
    - Multi-format output (PDF, HTML, Excel, etc.)
    - Real-time insights and recommendations
    - Performance benchmarking and comparison
    - Predictive analytics integration
    - Customizable templates and styling
    - Data validation and quality checks
    - Integration with all system components
    - Executive summaries and technical reports
    - Collaboration and decision tracking
    """
    
    def __init__(self, params: ReportParams = None):
        self.p = params or ReportParams()
        
        # Report storage
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Database for report metadata
        self.db_path = self.reports_dir / "reports.db"
        self._init_database()
        
        # Report templates
        self.templates = {}
        self._load_templates()
        
        # Generated insights
        self.insights = []
        self.insight_lock = threading.Lock()
        
        # Data sources
        self.data_sources = {}
        
        # Report generation queue
        self.generation_queue = []
        self.queue_lock = threading.Lock()
        
        # Background tasks
        self.background_tasks = []
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize report database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                file_path TEXT NOT NULL,
                format TEXT NOT NULL,
                size_bytes INTEGER,
                metadata TEXT
            )
        ''')
        
        # Create insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                insight_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                confidence REAL NOT NULL,
                impact_score REAL NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP,
                tags TEXT,
                data_points TEXT,
                recommendations TEXT
            )
        ''')
        
        # Create report_insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report_insights (
                report_id TEXT,
                insight_id TEXT,
                PRIMARY KEY (report_id, insight_id),
                FOREIGN KEY (report_id) REFERENCES reports (report_id),
                FOREIGN KEY (insight_id) REFERENCES insights (insight_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_templates(self):
        """Load report templates."""
        # HTML template for race summary
        self.templates[ReportType.RACE_SUMMARY] = {
            'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #e10600; color: white; padding: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
                    .chart {{ margin: 20px 0; }}
                    .insight {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; }}
                    .critical {{ border-left: 5px solid #e10600; }}
                    .high {{ border-left: 5px solid #ff6b00; }}
                    .medium {{ border-left: 5px solid #ffa500; }}
                    .low {{ border-left: 5px solid #32cd32; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>Race Overview</h2>
                    {race_overview}
                </div>
                
                <div class="section">
                    <h2>Key Metrics</h2>
                    {key_metrics}
                </div>
                
                <div class="section">
                    <h2>Insights</h2>
                    {insights}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {recommendations}
                </div>
            </body>
            </html>
            '''
        }
        
        # Template for tire analysis
        self.templates[ReportType.TIRE_ANALYSIS] = {
            'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #e10600; color: white; padding: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .tire-data {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
                    .tire-corner {{ padding: 15px; border: 1px solid #ccc; text-align: center; }}
                    .wear-level {{ font-size: 24px; font-weight: bold; }}
                    .temperature {{ font-size: 18px; }}
                    .chart {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>Tire Status</h2>
                    {tire_status}
                </div>
                
                <div class="section">
                    <h2>Wear Analysis</h2>
                    {wear_analysis}
                </div>
                
                <div class="section">
                    <h2>Temperature Analysis</h2>
                    {temperature_analysis}
                </div>
                
                <div class="section">
                    <h2>Predictions</h2>
                    {predictions}
                </div>
            </body>
            </html>
            '''
        }
        
        # Template for performance analysis
        self.templates[ReportType.PERFORMANCE_ANALYSIS] = {
            'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #e10600; color: white; padding: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Metrics</h2>
                    {performance_metrics}
                </div>
                
                <div class="section">
                    <h2>Sector Analysis</h2>
                    {sector_analysis}
                </div>
            </body>
            </html>
            '''
        }
        
        # Template for predictive insights
        self.templates[ReportType.PREDICTIVE_INSIGHTS] = {
            'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #e10600; color: white; padding: 20px; }}
                    .section {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>Degradation Prediction</h2>
                    {degradation_prediction}
                </div>
                
                <div class="section">
                    <h2>Pit Window Prediction</h2>
                    {pit_window_prediction}
                </div>
                
                <div class="section">
                    <h2>Weather Prediction</h2>
                    {weather_prediction}
                </div>
            </body>
            </html>
            '''
        }
    
    def _start_background_tasks(self):
        """Start background tasks for automated report generation."""
        def generate_reports():
            while True:
                time.sleep(self.p.generation_interval)
                
                # Generate automated reports
                self._generate_automated_reports()
                
                # Clean up expired insights
                self._cleanup_expired_insights()
        
        # Start background thread
        background_thread = threading.Thread(target=generate_reports)
        background_thread.daemon = True
        background_thread.start()
    
    def _generate_automated_reports(self):
        """Generate automated reports."""
        if not self.p.auto_generate:
            return
        
        # Generate race summary if race is active
        if self._is_race_active():
            self.generate_race_summary()
        
        # Generate tire analysis
        self.generate_tire_analysis()
        
        # Generate performance analysis
        self.generate_performance_analysis()
        
        # Generate predictive insights
        self.generate_predictive_insights()
    
    def _is_race_active(self) -> bool:
        """Check if race is currently active."""
        # This would check with the simulation engine
        return True  # Placeholder
    
    def generate_race_summary(self) -> str:
        """Generate race summary report."""
        report_id = str(uuid.uuid4())
        
        # Collect race data
        race_data = self._collect_race_data()
        
        # Generate insights
        insights = self._generate_race_insights(race_data)
        
        # Create report content
        report_content = self._create_race_summary_content(race_data, insights)
        
        # Generate report file
        file_path = self._generate_html_report(
            report_id, 
            "Race Summary", 
            report_content,
            ReportType.RACE_SUMMARY
        )
        
        # Store report metadata
        self._store_report_metadata(report_id, ReportType.RACE_SUMMARY, "Race Summary", file_path)
        
        return report_id
    
    def generate_tire_analysis(self) -> str:
        """Generate tire analysis report."""
        report_id = str(uuid.uuid4())
        
        # Collect tire data
        tire_data = self._collect_tire_data()
        
        # Generate insights
        insights = self._generate_tire_insights(tire_data)
        
        # Create report content
        report_content = self._create_tire_analysis_content(tire_data, insights)
        
        # Generate report file
        file_path = self._generate_html_report(
            report_id, 
            "Tire Analysis", 
            report_content,
            ReportType.TIRE_ANALYSIS
        )
        
        # Store report metadata
        self._store_report_metadata(report_id, ReportType.TIRE_ANALYSIS, "Tire Analysis", file_path)
        
        return report_id
    
    def generate_performance_analysis(self) -> str:
        """Generate performance analysis report."""
        report_id = str(uuid.uuid4())
        
        # Collect performance data
        performance_data = self._collect_performance_data()
        
        # Generate insights
        insights = self._generate_performance_insights(performance_data)
        
        # Create report content
        report_content = self._create_performance_analysis_content(performance_data, insights)
        
        # Generate report file
        file_path = self._generate_html_report(
            report_id, 
            "Performance Analysis", 
            report_content,
            ReportType.PERFORMANCE_ANALYSIS
        )
        
        # Store report metadata
        self._store_report_metadata(report_id, ReportType.PERFORMANCE_ANALYSIS, "Performance Analysis", file_path)
        
        return report_id
    
    def generate_predictive_insights(self) -> str:
        """Generate predictive insights report."""
        report_id = str(uuid.uuid4())
        
        # Collect predictive data
        predictive_data = self._collect_predictive_data()
        
        # Generate insights
        insights = self._generate_predictive_insights(predictive_data)
        
        # Create report content
        report_content = self._create_predictive_insights_content(predictive_data, insights)
        
        # Generate report file
        file_path = self._generate_html_report(
            report_id, 
            "Predictive Insights", 
            report_content,
            ReportType.PREDICTIVE_INSIGHTS
        )
        
        # Store report metadata
        self._store_report_metadata(report_id, ReportType.PREDICTIVE_INSIGHTS, "Predictive Insights", file_path)
        
        return report_id
    
    def _collect_race_data(self) -> Dict[str, Any]:
        """Collect race data from various sources."""
        # This would integrate with the simulation engine and other components
        return {
            'current_lap': 25,
            'total_laps': 58,
            'race_position': 3,
            'gap_to_leader': 12.5,
            'gap_to_next': 2.1,
            'tire_age': 15,
            'fuel_level': 0.6,
            'weather': {
                'temperature': 28.5,
                'humidity': 0.65,
                'rain_probability': 0.2
            },
            'lap_times': [89.2, 88.9, 89.1, 88.8, 89.0],
            'sector_times': {
                'sector1': [28.5, 28.3, 28.4, 28.2, 28.3],
                'sector2': [30.2, 30.0, 30.1, 29.9, 30.0],
                'sector3': [30.5, 30.6, 30.6, 30.7, 30.7]
            }
        }
    
    def _collect_tire_data(self) -> Dict[str, Any]:
        """Collect tire data from various sources."""
        # This would integrate with the thermal model and wear model
        return {
            'corners': {
                'FL': {'wear': 0.35, 'temperature': 95.2, 'pressure': 1.8},
                'FR': {'wear': 0.32, 'temperature': 97.1, 'pressure': 1.8},
                'RL': {'wear': 0.28, 'temperature': 92.8, 'pressure': 1.7},
                'RR': {'wear': 0.30, 'temperature': 94.5, 'pressure': 1.7}
            },
            'compound': 'C3',
            'age': 15,
            'predicted_life': 25,
            'grip_degradation': 0.15,
            'stiffness_reduction': 0.08
        }
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect performance data from various sources."""
        # This would integrate with the performance benchmarking component
        return {
            'lap_times': [89.2, 88.9, 89.1, 88.8, 89.0],
            'sector_times': {
                'sector1': [28.5, 28.3, 28.4, 28.2, 28.3],
                'sector2': [30.2, 30.0, 30.1, 29.9, 30.0],
                'sector3': [30.5, 30.6, 30.6, 30.7, 30.7]
            },
            'top_speed': 320.5,
            'average_speed': 185.2,
            'fuel_consumption': 2.8,
            'efficiency_score': 0.85
        }
    
    def _collect_predictive_data(self) -> Dict[str, Any]:
        """Collect predictive data from various sources."""
        # This would integrate with the predictive analytics component
        return {
            'tire_degradation_prediction': {
                'next_10_laps': [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53, 0.56, 0.59, 0.62],
                'confidence': 0.85
            },
            'pit_window_prediction': {
                'optimal_lap': 32,
                'confidence': 0.78
            },
            'weather_prediction': {
                'rain_probability': [0.2, 0.3, 0.4, 0.5, 0.6],
                'confidence': 0.70
            }
        }
    
    def _generate_race_insights(self, race_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from race data."""
        insights = []
        
        # Analyze lap times
        lap_times = race_data['lap_times']
        if len(lap_times) >= 3:
            recent_trend = statistics.mean(lap_times[-3:]) - statistics.mean(lap_times[-5:])
            
            if recent_trend > 0.5:
                insight = Insight(
                    insight_id=str(uuid.uuid4()),
                    title="Lap Time Degradation Detected",
                    description=f"Recent lap times show degradation of {recent_trend:.2f}s per lap",
                    category=InsightCategory.PERFORMANCE,
                    priority=Priority.HIGH,
                    confidence=0.8,
                    impact_score=0.7,
                    data_points=[{'metric': 'lap_time_trend', 'value': recent_trend}],
                    recommendations=["Consider pit stop strategy", "Check tire wear levels"],
                    created_at=datetime.now()
                )
                insights.append(insight)
        
        # Analyze tire age
        tire_age = race_data['tire_age']
        if tire_age > 20:
            insight = Insight(
                insight_id=str(uuid.uuid4()),
                title="Tire Age Warning",
                description=f"Tires are {tire_age} laps old, approaching optimal pit window",
                category=InsightCategory.STRATEGY,
                priority=Priority.MEDIUM,
                confidence=0.9,
                impact_score=0.6,
                data_points=[{'metric': 'tire_age', 'value': tire_age}],
                recommendations=["Plan pit stop strategy", "Monitor tire performance"],
                created_at=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _generate_tire_insights(self, tire_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from tire data."""
        insights = []
        
        # Analyze wear levels
        corners = tire_data['corners']
        max_wear = max(corner['wear'] for corner in corners.values())
        
        if max_wear > 0.4:
            insight = Insight(
                insight_id=str(uuid.uuid4()),
                title="High Tire Wear Detected",
                description=f"Maximum tire wear is {max_wear:.2f}, approaching critical levels",
                category=InsightCategory.SAFETY,
                priority=Priority.HIGH,
                confidence=0.9,
                impact_score=0.8,
                data_points=[{'metric': 'max_wear', 'value': max_wear}],
                recommendations=["Consider pit stop", "Reduce aggressive driving"],
                created_at=datetime.now()
            )
            insights.append(insight)
        
        # Analyze temperature distribution
        temperatures = [corner['temperature'] for corner in corners.values()]
        temp_variance = statistics.variance(temperatures)
        
        if temp_variance > 10:
            insight = Insight(
                insight_id=str(uuid.uuid4()),
                title="Tire Temperature Imbalance",
                description=f"Tire temperature variance is {temp_variance:.1f}°C, indicating imbalance",
                category=InsightCategory.TECHNICAL,
                priority=Priority.MEDIUM,
                confidence=0.7,
                impact_score=0.5,
                data_points=[{'metric': 'temp_variance', 'value': temp_variance}],
                recommendations=["Check tire pressures", "Adjust suspension settings"],
                created_at=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _generate_performance_insights(self, performance_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from performance data."""
        insights = []
        
        # Analyze sector performance
        sector_times = performance_data['sector_times']
        sector_averages = {sector: statistics.mean(times) for sector, times in sector_times.items()}
        
        # Find weakest sector
        weakest_sector = max(sector_averages, key=sector_averages.get)
        strongest_sector = min(sector_averages, key=sector_averages.get)
        
        insight = Insight(
            insight_id=str(uuid.uuid4()),
            title="Sector Performance Analysis",
            description=f"Weakest sector: {weakest_sector} ({sector_averages[weakest_sector]:.2f}s), Strongest: {strongest_sector} ({sector_averages[strongest_sector]:.2f}s)",
            category=InsightCategory.PERFORMANCE,
            priority=Priority.MEDIUM,
            confidence=0.8,
            impact_score=0.6,
            data_points=[{'metric': 'sector_averages', 'value': sector_averages}],
            recommendations=[f"Focus on {weakest_sector} improvement", "Maintain strength in other sectors"],
            created_at=datetime.now()
        )
        insights.append(insight)
        
        return insights
    
    def _generate_predictive_insights(self, predictive_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from predictive data."""
        insights = []
        
        # Analyze tire degradation prediction
        degradation_pred = predictive_data['tire_degradation_prediction']
        next_5_laps = degradation_pred['next_10_laps'][:5]
        
        if max(next_5_laps) > 0.5:
            insight = Insight(
                insight_id=str(uuid.uuid4()),
                title="Tire Degradation Prediction",
                description=f"Predicted tire wear will exceed 50% in next 5 laps",
                category=InsightCategory.PREDICTIVE,
                priority=Priority.HIGH,
                confidence=degradation_pred['confidence'],
                impact_score=0.8,
                data_points=[{'metric': 'predicted_wear', 'value': next_5_laps}],
                recommendations=["Plan pit stop strategy", "Monitor tire performance closely"],
                created_at=datetime.now()
            )
            insights.append(insight)
        
        # Analyze pit window prediction
        pit_pred = predictive_data['pit_window_prediction']
        current_lap = 25  # This would come from race data
        optimal_lap = pit_pred['optimal_lap']
        
        if abs(current_lap - optimal_lap) <= 2:
            insight = Insight(
                insight_id=str(uuid.uuid4()),
                title="Optimal Pit Window",
                description=f"Current lap {current_lap} is within optimal pit window (lap {optimal_lap})",
                category=InsightCategory.STRATEGY,
                priority=Priority.HIGH,
                confidence=pit_pred['confidence'],
                impact_score=0.7,
                data_points=[{'metric': 'optimal_pit_lap', 'value': optimal_lap}],
                recommendations=["Consider pit stop now", "Monitor traffic conditions"],
                created_at=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _create_race_summary_content(self, race_data: Dict[str, Any], insights: List[Insight]) -> Dict[str, str]:
        """Create race summary report content."""
        # Race overview
        race_overview = f"""
        <div class="metric">Current Lap: {race_data['current_lap']}</div>
        <div class="metric">Race Position: {race_data['race_position']}</div>
        <div class="metric">Gap to Leader: {race_data['gap_to_leader']}s</div>
        <div class="metric">Tire Age: {race_data['tire_age']} laps</div>
        <div class="metric">Fuel Level: {race_data['fuel_level']:.1%}</div>
        """
        
        # Key metrics
        key_metrics = f"""
        <div class="metric">Average Lap Time: {statistics.mean(race_data['lap_times']):.2f}s</div>
        <div class="metric">Best Lap Time: {min(race_data['lap_times']):.2f}s</div>
        <div class="metric">Top Speed: {race_data.get('top_speed', 'N/A')} km/h</div>
        <div class="metric">Weather: {race_data['weather']['temperature']:.1f}°C, {race_data['weather']['humidity']:.1%} humidity</div>
        """
        
        # Insights
        insights_html = ""
        for insight in insights:
            priority_class = insight.priority.value
            insights_html += f"""
            <div class="insight {priority_class}">
                <h3>{insight.title}</h3>
                <p>{insight.description}</p>
                <p><strong>Confidence:</strong> {insight.confidence:.1%}</p>
                <p><strong>Impact:</strong> {insight.impact_score:.1%}</p>
            </div>
            """
        
        # Recommendations
        recommendations_html = ""
        for insight in insights:
            for rec in insight.recommendations:
                recommendations_html += f"<li>{rec}</li>"
        
        if recommendations_html:
            recommendations_html = f"<ul>{recommendations_html}</ul>"
        
        return {
            'race_overview': race_overview,
            'key_metrics': key_metrics,
            'insights': insights_html,
            'recommendations': recommendations_html
        }
    
    def _create_tire_analysis_content(self, tire_data: Dict[str, Any], insights: List[Insight]) -> Dict[str, str]:
        """Create tire analysis report content."""
        # Tire status
        tire_status = ""
        for corner, data in tire_data['corners'].items():
            tire_status += f"""
            <div class="tire-corner">
                <h3>{corner}</h3>
                <div class="wear-level">{data['wear']:.1%}</div>
                <div class="temperature">{data['temperature']:.1f}°C</div>
                <div class="pressure">{data['pressure']:.1f} bar</div>
            </div>
            """
        
        # Wear analysis
        wear_analysis = f"""
        <p>Compound: {tire_data['compound']}</p>
        <p>Age: {tire_data['age']} laps</p>
        <p>Predicted Life: {tire_data['predicted_life']} laps</p>
        <p>Grip Degradation: {tire_data['grip_degradation']:.1%}</p>
        <p>Stiffness Reduction: {tire_data['stiffness_reduction']:.1%}</p>
        """
        
        # Temperature analysis
        temperatures = [data['temperature'] for data in tire_data['corners'].values()]
        temp_analysis = f"""
        <p>Average Temperature: {statistics.mean(temperatures):.1f}°C</p>
        <p>Temperature Range: {max(temperatures) - min(temperatures):.1f}°C</p>
        <p>Maximum Temperature: {max(temperatures):.1f}°C</p>
        <p>Minimum Temperature: {min(temperatures):.1f}°C</p>
        """
        
        # Predictions
        predictions = f"""
        <p>Predicted pit window: {tire_data['predicted_life'] - tire_data['age']} laps remaining</p>
        <p>Risk of tire failure: {'High' if max(data['wear'] for data in tire_data['corners'].values()) > 0.5 else 'Low'}</p>
        """
        
        return {
            'tire_status': tire_status,
            'wear_analysis': wear_analysis,
            'temperature_analysis': temp_analysis,
            'predictions': predictions
        }
    
    def _create_performance_analysis_content(self, performance_data: Dict[str, Any], insights: List[Insight]) -> Dict[str, str]:
        """Create performance analysis report content."""
        # Performance metrics
        metrics = f"""
        <div class="metric">Average Lap Time: {statistics.mean(performance_data['lap_times']):.2f}s</div>
        <div class="metric">Best Lap Time: {min(performance_data['lap_times']):.2f}s</div>
        <div class="metric">Top Speed: {performance_data['top_speed']} km/h</div>
        <div class="metric">Average Speed: {performance_data['average_speed']} km/h</div>
        <div class="metric">Fuel Consumption: {performance_data['fuel_consumption']} L/lap</div>
        <div class="metric">Efficiency Score: {performance_data['efficiency_score']:.1%}</div>
        """
        
        # Sector analysis
        sector_times = performance_data['sector_times']
        sector_analysis = ""
        for sector, times in sector_times.items():
            avg_time = statistics.mean(times)
            best_time = min(times)
            sector_analysis += f"""
            <div class="metric">
                <h3>{sector}</h3>
                <p>Average: {avg_time:.2f}s</p>
                <p>Best: {best_time:.2f}s</p>
            </div>
            """
        
        return {
            'performance_metrics': metrics,
            'sector_analysis': sector_analysis
        }
    
    def _create_predictive_insights_content(self, predictive_data: Dict[str, Any], insights: List[Insight]) -> Dict[str, str]:
        """Create predictive insights report content."""
        # Tire degradation prediction
        degradation_pred = predictive_data['tire_degradation_prediction']
        degradation_html = f"""
        <p>Predicted wear over next 10 laps:</p>
        <ul>
        """
        for i, wear in enumerate(degradation_pred['next_10_laps']):
            degradation_html += f"<li>Lap {i+1}: {wear:.1%}</li>"
        degradation_html += f"</ul><p>Confidence: {degradation_pred['confidence']:.1%}</p>"
        
        # Pit window prediction
        pit_pred = predictive_data['pit_window_prediction']
        pit_html = f"""
        <p>Optimal pit window: Lap {pit_pred['optimal_lap']}</p>
        <p>Confidence: {pit_pred['confidence']:.1%}</p>
        """
        
        # Weather prediction
        weather_pred = predictive_data['weather_prediction']
        weather_html = f"""
        <p>Rain probability over next 5 laps:</p>
        <ul>
        """
        for i, prob in enumerate(weather_pred['rain_probability']):
            weather_html += f"<li>Lap {i+1}: {prob:.1%}</li>"
        weather_html += f"</ul><p>Confidence: {weather_pred['confidence']:.1%}</p>"
        
        return {
            'degradation_prediction': degradation_html,
            'pit_window_prediction': pit_html,
            'weather_prediction': weather_html
        }
    
    def _generate_html_report(self, report_id: str, title: str, content: Dict[str, str], 
                             report_type: ReportType) -> str:
        """Generate HTML report file."""
        # Get template
        template = self.templates[report_type]['html']
        
        # Format content
        formatted_content = template.format(
            title=title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **content
        )
        
        # Save file
        file_path = self.reports_dir / f"{report_id}_{report_type.value}.html"
        with open(file_path, 'w') as f:
            f.write(formatted_content)
        
        return str(file_path)
    
    def _store_report_metadata(self, report_id: str, report_type: ReportType, 
                              title: str, file_path: str):
        """Store report metadata in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get file size
        file_size = Path(file_path).stat().st_size
        
        # Insert report record
        cursor.execute('''
            INSERT INTO reports (report_id, report_type, title, created_at, file_path, format, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (report_id, report_type.value, title, datetime.now(), file_path, 'html', file_size))
        
        conn.commit()
        conn.close()
    
    def _cleanup_expired_insights(self):
        """Clean up expired insights."""
        current_time = datetime.now()
        
        with self.insight_lock:
            expired_insights = [
                insight for insight in self.insights
                if insight.expires_at and insight.expires_at < current_time
            ]
            
            for insight in expired_insights:
                self.insights.remove(insight)
    
    def get_report_list(self, report_type: ReportType = None) -> List[Dict[str, Any]]:
        """Get list of generated reports."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if report_type:
            cursor.execute('''
                SELECT report_id, report_type, title, created_at, file_path, format, size_bytes
                FROM reports
                WHERE report_type = ?
                ORDER BY created_at DESC
            ''', (report_type.value,))
        else:
            cursor.execute('''
                SELECT report_id, report_type, title, created_at, file_path, format, size_bytes
                FROM reports
                ORDER BY created_at DESC
            ''')
        
        reports = []
        for row in cursor.fetchall():
            reports.append({
                'report_id': row[0],
                'report_type': row[1],
                'title': row[2],
                'created_at': row[3],
                'file_path': row[4],
                'format': row[5],
                'size_bytes': row[6]
            })
        
        conn.close()
        return reports
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of all insights."""
        with self.insight_lock:
            total_insights = len(self.insights)
            
            # Group by priority
            priority_counts = defaultdict(int)
            for insight in self.insights:
                priority_counts[insight.priority] += 1
            
            # Group by category
            category_counts = defaultdict(int)
            for insight in self.insights:
                category_counts[insight.category] += 1
            
            # Calculate average confidence
            avg_confidence = statistics.mean([insight.confidence for insight in self.insights]) if self.insights else 0
            
            # Calculate average impact
            avg_impact = statistics.mean([insight.impact_score for insight in self.insights]) if self.insights else 0
            
            return {
                'total_insights': total_insights,
                'priority_distribution': dict(priority_counts),
                'category_distribution': dict(category_counts),
                'average_confidence': avg_confidence,
                'average_impact': avg_impact,
                'recent_insights': self.insights[-10:] if self.insights else []
            }
    
    def export_report(self, report_id: str, format: ReportFormat) -> str:
        """Export report in specified format."""
        # This would implement format conversion
        # For now, return the HTML file path
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT file_path FROM reports WHERE report_id = ?', (report_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        else:
            raise ValueError("Report not found")
    
    def __del__(self):
        """Cleanup resources."""
        pass
