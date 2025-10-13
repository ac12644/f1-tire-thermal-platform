from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime, timedelta
import json

class VisualizationType(Enum):
    """Types of visualizations."""
    THERMAL_HEATMAP = "thermal_heatmap"
    TIRE_3D_MODEL = "tire_3d_model"
    WEAR_DISTRIBUTION = "wear_distribution"
    TEMPERATURE_EVOLUTION = "temperature_evolution"
    PERFORMANCE_MATRIX = "performance_matrix"
    DRIVER_COMPARISON = "driver_comparison"
    STRATEGY_ANALYSIS = "strategy_analysis"
    WEATHER_IMPACT = "weather_impact"

class ChartType(Enum):
    """Types of charts."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    CONTOUR_PLOT = "contour_plot"
    SURFACE_PLOT = "surface_plot"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"

@dataclass
class AdvancedVisualizationParams:
    """Parameters for advanced visualization."""
    # Visualization parameters
    figure_size: Tuple[int, int] = (1200, 800)
    dpi: int = 300
    color_scheme: str = "viridis"
    
    # 3D visualization parameters
    tire_radius: float = 0.33  # meters
    tire_width: float = 0.245   # meters
    resolution: int = 50        # Resolution for 3D models
    
    # Heat map parameters
    heatmap_resolution: int = 100
    temperature_range: Tuple[float, float] = (60.0, 120.0)
    wear_range: Tuple[float, float] = (0.0, 1.0)
    
    # Animation parameters
    animation_enabled: bool = True
    animation_duration: int = 1000
    animation_frames: int = 50
    
    # Export parameters
    export_formats: List[str] = None
    export_quality: str = "high"
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["png", "svg", "html"]

class AdvancedVisualization:
    """
    Advanced visualization system for F1 tire temperature management.
    
    Features:
    - 3D tire modeling with thermal visualization
    - Interactive heat maps for temperature and wear distribution
    - Real-time temperature evolution charts
    - Performance matrix visualizations
    - Driver comparison charts
    - Strategy analysis visualizations
    - Weather impact charts
    - Animated visualizations for time series data
    """
    
    def __init__(self, params: AdvancedVisualizationParams = None):
        self.p = params or AdvancedVisualizationParams()
        
        # Visualization data
        self.visualization_data = {}
        self.chart_templates = {}
        
        # 3D model data
        self.tire_3d_models = {}
        self.thermal_meshes = {}
        
        # Animation data
        self.animation_frames = {}
        self.animation_data = {}
        
        # Export settings
        self.export_settings = {
            'png': {'dpi': self.p.dpi, 'bbox_inches': 'tight'},
            'svg': {'format': 'svg'},
            'html': {'include_plotlyjs': True}
        }
    
    def create_thermal_heatmap(self, thermal_data: Dict[str, Any], 
                              visualization_type: str = "temperature") -> go.Figure:
        """
        Create thermal heat map visualization.
        
        Args:
            thermal_data: Thermal data dictionary
            visualization_type: Type of heat map (temperature, wear, pressure)
            
        Returns:
            Plotly figure object
        """
        # Extract thermal data
        tread_temp = thermal_data.get('tread_temp', 0)
        carcass_temp = thermal_data.get('carcass_temp', 0)
        rim_temp = thermal_data.get('rim_temp', 0)
        
        # Create coordinate grid
        x = np.linspace(-self.p.tire_width/2, self.p.tire_width/2, self.p.heatmap_resolution)
        y = np.linspace(-self.p.tire_radius, self.p.tire_radius, self.p.heatmap_resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create temperature distribution
        if visualization_type == "temperature":
            # Create radial temperature distribution
            Z = self._create_radial_temperature_distribution(X, Y, tread_temp, carcass_temp, rim_temp)
            colorbar_title = "Temperature (°C)"
            colorscale = "thermal"
        elif visualization_type == "wear":
            # Create wear distribution
            Z = self._create_wear_distribution(X, Y, thermal_data.get('wear_level', 0))
            colorbar_title = "Wear Level"
            colorscale = "RdYlBu_r"
        elif visualization_type == "pressure":
            # Create pressure distribution
            Z = self._create_pressure_distribution(X, Y, thermal_data.get('pressure', 1.5))
            colorbar_title = "Pressure (bar)"
            colorscale = "blues"
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        # Create heat map
        fig = go.Figure(data=go.Heatmap(
            z=Z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            hovertemplate=f"{visualization_type.title()}: %{{z:.2f}}<br>" +
                         "X: %{x:.3f}<br>" +
                         "Y: %{y:.3f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Tire {visualization_type.title()} Heat Map",
            xaxis_title="Width (m)",
            yaxis_title="Radius (m)",
            width=self.p.figure_size[0],
            height=self.p.figure_size[1],
            font=dict(size=12)
        )
        
        return fig
    
    def _create_radial_temperature_distribution(self, X: np.ndarray, Y: np.ndarray, 
                                              tread_temp: float, carcass_temp: float, 
                                              rim_temp: float) -> np.ndarray:
        """Create radial temperature distribution for tire."""
        # Calculate distance from center
        R = np.sqrt(X**2 + Y**2)
        
        # Create temperature zones
        Z = np.zeros_like(R)
        
        # Tread zone (outer region)
        tread_mask = R > 0.8 * self.p.tire_radius
        Z[tread_mask] = tread_temp
        
        # Carcass zone (middle region)
        carcass_mask = (R > 0.4 * self.p.tire_radius) & (R <= 0.8 * self.p.tire_radius)
        Z[carcass_mask] = carcass_temp
        
        # Rim zone (inner region)
        rim_mask = R <= 0.4 * self.p.tire_radius
        Z[rim_mask] = rim_temp
        
        # Smooth transitions
        Z = self._smooth_temperature_transitions(Z, R)
        
        return Z
    
    def _create_wear_distribution(self, X: np.ndarray, Y: np.ndarray, 
                                wear_level: float) -> np.ndarray:
        """Create wear distribution for tire."""
        # Calculate distance from center
        R = np.sqrt(X**2 + Y**2)
        
        # Create wear distribution (higher wear at edges)
        Z = np.zeros_like(R)
        
        # Maximum wear at tire edges
        edge_mask = R > 0.9 * self.p.tire_radius
        Z[edge_mask] = wear_level
        
        # Gradual wear decrease towards center
        center_mask = R <= 0.9 * self.p.tire_radius
        Z[center_mask] = wear_level * (R[center_mask] / (0.9 * self.p.tire_radius))
        
        return Z
    
    def _create_pressure_distribution(self, X: np.ndarray, Y: np.ndarray, 
                                    pressure: float) -> np.ndarray:
        """Create pressure distribution for tire."""
        # Calculate distance from center
        R = np.sqrt(X**2 + Y**2)
        
        # Create pressure distribution (higher pressure at center)
        Z = np.zeros_like(R)
        
        # Maximum pressure at center
        center_mask = R <= 0.2 * self.p.tire_radius
        Z[center_mask] = pressure
        
        # Gradual pressure decrease towards edges
        edge_mask = R > 0.2 * self.p.tire_radius
        Z[edge_mask] = pressure * (1 - (R[edge_mask] - 0.2 * self.p.tire_radius) / (0.8 * self.p.tire_radius))
        
        return Z
    
    def _smooth_temperature_transitions(self, Z: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Smooth temperature transitions between zones."""
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        Z_smoothed = gaussian_filter(Z, sigma=1.0)
        
        return Z_smoothed
    
    def create_tire_3d_model(self, thermal_data: Dict[str, Any], 
                            wear_data: Dict[str, Any]) -> go.Figure:
        """
        Create 3D tire model with thermal and wear visualization.
        
        Args:
            thermal_data: Thermal data dictionary
            wear_data: Wear data dictionary
            
        Returns:
            Plotly figure object
        """
        # Create tire geometry
        theta = np.linspace(0, 2*np.pi, self.p.resolution)
        z = np.linspace(-self.p.tire_width/2, self.p.tire_width/2, self.p.resolution)
        Theta, Z = np.meshgrid(theta, z)
        
        # Create tire surface
        R = self.p.tire_radius
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        # Create temperature surface
        tread_temp = thermal_data.get('tread_temp', 0)
        carcass_temp = thermal_data.get('carcass_temp', 0)
        rim_temp = thermal_data.get('rim_temp', 0)
        
        # Create temperature distribution on tire surface
        T = np.zeros_like(X)
        
        # Tread temperature (outer surface)
        T[:, :] = tread_temp
        
        # Add temperature variation based on position
        T += 5 * np.sin(Theta) * np.cos(Z * 10)  # Simulate temperature variation
        
        # Create wear surface
        wear_level = wear_data.get('wear_level', 0)
        W = np.full_like(X, wear_level)
        
        # Add wear variation
        W += 0.1 * np.sin(Theta * 2) * np.cos(Z * 5)  # Simulate wear variation
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Add temperature surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=T,
            colorscale="thermal",
            colorbar=dict(title="Temperature (°C)", x=0.02),
            name="Temperature",
            hovertemplate="Temperature: %{surfacecolor:.1f}°C<br>" +
                         "X: %{x:.3f}<br>" +
                         "Y: %{y:.3f}<br>" +
                         "Z: %{z:.3f}<extra></extra>"
        ))
        
        # Add wear surface (semi-transparent)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=W,
            colorscale="RdYlBu_r",
            colorbar=dict(title="Wear Level", x=0.98),
            name="Wear",
            opacity=0.7,
            hovertemplate="Wear: %{surfacecolor:.2f}<br>" +
                         "X: %{x:.3f}<br>" +
                         "Y: %{y:.3f}<br>" +
                         "Z: %{z:.3f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="3D Tire Model - Temperature and Wear Visualization",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data"
            ),
            width=self.p.figure_size[0],
            height=self.p.figure_size[1]
        )
        
        return fig
    
    def create_temperature_evolution(self, time_series_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create temperature evolution chart over time.
        
        Args:
            time_series_data: List of time series data points
            
        Returns:
            Plotly figure object
        """
        # Extract data
        timestamps = [data['timestamp'] for data in time_series_data]
        tread_temps = [data['tread_temp'] for data in time_series_data]
        carcass_temps = [data['carcass_temp'] for data in time_series_data]
        rim_temps = [data['rim_temp'] for data in time_series_data]
        
        # Create figure
        fig = go.Figure()
        
        # Add temperature traces
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=tread_temps,
            mode='lines+markers',
            name='Tread Temperature',
            line=dict(color='red', width=2),
            hovertemplate="Tread: %{y:.1f}°C<br>Time: %{x}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=carcass_temps,
            mode='lines+markers',
            name='Carcass Temperature',
            line=dict(color='orange', width=2),
            hovertemplate="Carcass: %{y:.1f}°C<br>Time: %{x}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=rim_temps,
            mode='lines+markers',
            name='Rim Temperature',
            line=dict(color='blue', width=2),
            hovertemplate="Rim: %{y:.1f}°C<br>Time: %{x}<extra></extra>"
        ))
        
        # Add optimal temperature bands
        fig.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.1, 
                     annotation_text="Optimal Range", annotation_position="top right")
        
        # Update layout
        fig.update_layout(
            title="Tire Temperature Evolution",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            width=self.p.figure_size[0],
            height=self.p.figure_size[1],
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_matrix(self, performance_data: Dict[str, Any]) -> go.Figure:
        """
        Create performance matrix visualization.
        
        Args:
            performance_data: Performance data dictionary
            
        Returns:
            Plotly figure object
        """
        # Extract performance metrics
        drivers = list(performance_data.keys())
        metrics = ['lap_time', 'tire_life', 'thermal_stability', 'wear_rate', 
                  'consistency_score', 'adaptation_time', 'strategy_success']
        
        # Create performance matrix
        performance_matrix = np.zeros((len(drivers), len(metrics)))
        
        for i, driver in enumerate(drivers):
            for j, metric in enumerate(metrics):
                if metric in performance_data[driver]:
                    performance_matrix[i, j] = performance_data[driver][metric]
        
        # Create heat map
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=metrics,
            y=drivers,
            colorscale="RdYlGn",
            colorbar=dict(title="Performance Score"),
            hovertemplate="Driver: %{y}<br>" +
                         "Metric: %{x}<br>" +
                         "Score: %{z:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Driver Performance Matrix",
            xaxis_title="Performance Metrics",
            yaxis_title="Drivers",
            width=self.p.figure_size[0],
            height=self.p.figure_size[1]
        )
        
        return fig
    
    def create_driver_comparison(self, driver_data: Dict[str, Any]) -> go.Figure:
        """
        Create driver comparison visualization.
        
        Args:
            driver_data: Driver comparison data
            
        Returns:
            Plotly figure object
        """
        # Extract data
        drivers = list(driver_data.keys())
        metrics = ['lap_time', 'tire_life', 'thermal_stability', 'wear_rate', 
                  'consistency_score', 'adaptation_time', 'strategy_success']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=metrics,
            specs=[[{"type": "bar"} for _ in range(4)],
                   [{"type": "bar"} for _ in range(4)]]
        )
        
        # Add traces for each metric
        for i, metric in enumerate(metrics):
            row = i // 4 + 1
            col = i % 4 + 1
            
            values = [driver_data[driver].get(metric, 0) for driver in drivers]
            
            fig.add_trace(
                go.Bar(
                    x=drivers,
                    y=values,
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Driver Performance Comparison",
            width=self.p.figure_size[0],
            height=self.p.figure_size[1]
        )
        
        return fig
    
    def create_strategy_analysis(self, strategy_data: Dict[str, Any]) -> go.Figure:
        """
        Create strategy analysis visualization.
        
        Args:
            strategy_data: Strategy analysis data
            
        Returns:
            Plotly figure object
        """
        # Extract strategy data
        strategies = list(strategy_data.keys())
        success_rates = [strategy_data[strategy]['success_rate'] for strategy in strategies]
        performance_gains = [strategy_data[strategy]['performance_gain'] for strategy in strategies]
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=success_rates,
            y=performance_gains,
            mode='markers+text',
            text=strategies,
            textposition="top center",
            marker=dict(
                size=20,
                color=success_rates,
                colorscale="RdYlGn",
                colorbar=dict(title="Success Rate")
            ),
            hovertemplate="Strategy: %{text}<br>" +
                         "Success Rate: %{x:.2f}<br>" +
                         "Performance Gain: %{y:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Strategy Analysis - Success Rate vs Performance Gain",
            xaxis_title="Success Rate",
            yaxis_title="Performance Gain",
            width=self.p.figure_size[0],
            height=self.p.figure_size[1]
        )
        
        return fig
    
    def create_weather_impact(self, weather_data: Dict[str, Any]) -> go.Figure:
        """
        Create weather impact visualization.
        
        Args:
            weather_data: Weather impact data
            
        Returns:
            Plotly figure object
        """
        # Extract weather data
        weather_conditions = list(weather_data.keys())
        temperature_impacts = [weather_data[condition]['temperature_impact'] for condition in weather_conditions]
        wear_impacts = [weather_data[condition]['wear_impact'] for condition in weather_conditions]
        performance_impacts = [weather_data[condition]['performance_impact'] for condition in weather_conditions]
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name="Temperature Impact",
            x=weather_conditions,
            y=temperature_impacts,
            marker_color="red"
        ))
        
        fig.add_trace(go.Bar(
            name="Wear Impact",
            x=weather_conditions,
            y=wear_impacts,
            marker_color="orange"
        ))
        
        fig.add_trace(go.Bar(
            name="Performance Impact",
            x=weather_conditions,
            y=performance_impacts,
            marker_color="blue"
        ))
        
        # Update layout
        fig.update_layout(
            title="Weather Impact on Tire Performance",
            xaxis_title="Weather Conditions",
            yaxis_title="Impact Score",
            barmode="group",
            width=self.p.figure_size[0],
            height=self.p.figure_size[1]
        )
        
        return fig
    
    def create_animated_visualization(self, time_series_data: List[Dict[str, Any]], 
                                    visualization_type: str = "temperature") -> go.Figure:
        """
        Create animated visualization for time series data.
        
        Args:
            time_series_data: Time series data
            visualization_type: Type of visualization
            
        Returns:
            Plotly figure object
        """
        if not self.p.animation_enabled:
            return self.create_temperature_evolution(time_series_data)
        
        # Create frames for animation
        frames = []
        
        for i, data in enumerate(time_series_data):
            frame = go.Frame(
                data=[go.Scatter(
                    x=[data['timestamp']],
                    y=[data['tread_temp']],
                    mode='markers',
                    marker=dict(size=10, color='red')
                )],
                name=f"frame_{i}"
            )
            frames.append(frame)
        
        # Create initial figure
        fig = go.Figure(
            data=[go.Scatter(
                x=[],
                y=[],
                mode='markers',
                marker=dict(size=10, color='red')
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Animated Temperature Evolution",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            width=self.p.figure_size[0],
            height=self.p.figure_size[1],
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": self.p.animation_duration, "redraw": True}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )
        
        return fig
    
    def export_visualization(self, fig: go.Figure, filename: str, 
                           format: str = "html") -> str:
        """
        Export visualization to file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Export format
            
        Returns:
            Path to exported file
        """
        if format == "html":
            return fig.write_html(filename)
        elif format == "png":
            return fig.write_image(filename, **self.export_settings['png'])
        elif format == "svg":
            return fig.write_image(filename, **self.export_settings['svg'])
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get comprehensive visualization summary."""
        return {
            'visualization_types': [vt.value for vt in VisualizationType],
            'chart_types': [ct.value for ct in ChartType],
            'figure_size': self.p.figure_size,
            'color_scheme': self.p.color_scheme,
            'animation_enabled': self.p.animation_enabled,
            'export_formats': self.p.export_formats,
            'visualization_data_count': len(self.visualization_data),
            'tire_3d_models_count': len(self.tire_3d_models),
            'animation_frames_count': len(self.animation_frames)
        }
