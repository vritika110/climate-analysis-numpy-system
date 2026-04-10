"""
COMPLETE CLIMATE ANALYSIS SYSTEM WITH NUMPY AND VISUALIZATIONS
=============================================================

This is a complete, runnable implementation of the comprehensive climate analysis project.
It demonstrates every major NumPy feature through realistic climate science applications,
enhanced with professional Seaborn/Matplotlib visualizations.

Purpose: Complete NumPy tutorial with scientific visualizations
Requirements: numpy, matplotlib, seaborn, pandas

Run this file to see the complete climate analysis system in action!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import warnings
from typing import Tuple, List, Dict, Optional
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# Configure warnings and plotting
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# STEP 1: CLIMATE DATA GENERATION
# ============================================================================

class ClimateDataGenerator:
    """Generate realistic climate data using NumPy array creation methods"""
    
    def __init__(self, years: int = 5, stations: int = 20):
        self.years = years
        self.stations = stations
        self.days_per_year = 365
        self.total_days = years * self.days_per_year
        
        # Set random seed for reproducibility
        np.random.seed(42)
        print(f"🌍 Climate Data Generator initialized")
        print(f"   Study period: {years} years ({self.total_days:,} days)")
        print(f"   Weather stations: {stations}")
        
    def generate_temperature_data(self) -> np.ndarray:
        """Generate realistic temperature data with seasonal patterns"""
        print("📊 Generating temperature data...")
        
        # Base temperature patterns
        base_temp = np.full((self.stations,), 15.0)  # Global average
        latitude_effect = np.linspace(-10, 10, self.stations)  # Latitude variation
        
        # Seasonal patterns using trigonometry
        days = np.arange(self.total_days)
        seasonal_pattern = 10 * np.sin(2 * np.pi * days / 365 + np.pi/2)
        
        # Create 3D array: (days, stations, measurements)
        temperature = np.zeros((self.total_days, self.stations, 3))
        
        # Generate temperature data with broadcasting
        for measurement in range(3):  # min, avg, max
            temp_adjustment = measurement * 5 - 5  # -5, 0, 5 degrees
            
            temperature[:, :, measurement] = (
                base_temp[np.newaxis, :] +  # Broadcasting
                latitude_effect[np.newaxis, :] +
                seasonal_pattern[:, np.newaxis] +
                np.random.normal(0, 2, (self.total_days, self.stations)) +
                temp_adjustment
            )
            
        print(f"   Temperature array shape: {temperature.shape}")
        print(f"   Temperature range: {temperature.min():.1f}°C to {temperature.max():.1f}°C")
        return temperature
    
    def generate_precipitation_data(self) -> np.ndarray:
        """Generate realistic precipitation data"""
        print("🌧️ Generating precipitation data...")
        
        # Precipitation follows gamma distribution
        precipitation = np.random.gamma(2, 2, (self.total_days, self.stations))
        
        # Add seasonal patterns
        days = np.arange(self.total_days)
        wetness_cycle = 1 + 0.5 * np.sin(2 * np.pi * days / 365)
        precipitation *= wetness_cycle[:, np.newaxis]
        
        # Add dry spells using boolean indexing
        dry_probability = 0.6
        is_dry = np.random.random((self.total_days, self.stations)) < dry_probability
        precipitation[is_dry] = 0
        
        print(f"   Precipitation array shape: {precipitation.shape}")
        print(f"   Average annual precipitation: {precipitation.sum(axis=0).mean():.0f}mm")
        return precipitation

# ============================================================================
# STEP 2: ADVANCED INDEXING AND DATA PROCESSING
# ============================================================================

class DataProcessor:
    """Advanced array manipulation and indexing techniques"""
    
    def __init__(self, temperature: np.ndarray, precipitation: np.ndarray):
        self.temperature = temperature
        self.precipitation = precipitation
        self.days, self.stations, self.temp_measurements = temperature.shape
        
    def demonstrate_indexing_techniques(self):
        """Comprehensive demonstration of NumPy indexing"""
        print("\n🔍 Advanced indexing techniques...")
        
        # Boolean indexing - find extreme weather
        hot_days = self.temperature[:, :, 1] > 30  # Average temperature > 30°C
        hot_day_count = hot_days.sum(axis=0)
        print(f"   Hot days per station (avg): {hot_day_count.mean():.1f}")
        
        # Fancy indexing - dynamic station selection
        n_stations_to_select = min(6, self.stations)
        station_indices = np.linspace(0, self.stations-1, n_stations_to_select, dtype=int)
        selected_stations = station_indices
        
        n_days_to_select = min(4, self.days)
        day_indices = np.linspace(0, min(self.days-1, 350), n_days_to_select, dtype=int)
        selected_days = day_indices
        
        subset = self.temperature[np.ix_(selected_days, selected_stations)]
        print(f"   Fancy indexed subset shape: {subset.shape}")
        
        # Complex boolean conditions
        extreme_weather = (
            (self.temperature[:, :, 1] > 35) |  # Very hot OR
            (self.temperature[:, :, 1] < -10) |  # Very cold OR
            (self.precipitation > 50)  # Heavy rain
        )
        extreme_days = extreme_weather.sum()
        print(f"   Total extreme weather events: {extreme_days}")
        
        return hot_days, extreme_weather
    
    def reshape_and_transpose_operations(self):
        """Demonstrate array reshaping and transposition"""
        print("\n🔄 Reshaping and transposition...")
        
        # Reshape for yearly analysis
        years = self.days // 365
        if years > 0:
            yearly_data = self.temperature[:years*365].reshape(years, 365, self.stations, 3)
            print(f"   Yearly reshaped data: {yearly_data.shape}")
        else:
            yearly_data = self.temperature
        
        # Transpose for station-first analysis
        station_first = self.temperature.transpose(1, 0, 2)
        print(f"   Station-first shape: {station_first.shape}")
        
        # Calculate temperature range
        temp_range = self.temperature[:, :, 2] - self.temperature[:, :, 0]  # max - min
        print(f"   Average daily temperature range: {temp_range.mean():.2f}°C")
        
        return yearly_data, temp_range

# ============================================================================
# STEP 3: MATHEMATICAL OPERATIONS AND STATISTICS
# ============================================================================

class MathematicalAnalyzer:
    """Comprehensive mathematical operations using NumPy"""
    
    def __init__(self, temperature: np.ndarray, precipitation: np.ndarray):
        self.temperature = temperature
        self.precipitation = precipitation
        
    def basic_mathematical_operations(self):
        """Demonstrate basic mathematical operations"""
        print("\n🧮 Basic mathematical operations...")
        
        temp_celsius = self.temperature[:, :, 1]  # Average temperatures
        
        # Temperature conversions using universal functions
        temp_fahrenheit = 9/5 * temp_celsius + 32
        temp_kelvin = temp_celsius + 273.15
        
        print(f"   Temperature ranges:")
        print(f"     Celsius: {temp_celsius.min():.1f} to {temp_celsius.max():.1f}")
        print(f"     Fahrenheit: {temp_fahrenheit.min():.1f} to {temp_fahrenheit.max():.1f}")
        print(f"     Kelvin: {temp_kelvin.min():.1f} to {temp_kelvin.max():.1f}")
        
        # Advanced mathematical functions
        temp_squared = np.square(temp_celsius)
        temp_sqrt = np.sqrt(np.abs(temp_celsius))
        
        # Trigonometric modeling
        days = np.arange(len(temp_celsius))
        seasonal_model = (
            np.sin(2 * np.pi * days / 365) * 10 +
            np.cos(2 * np.pi * days / 365 * 2) * 3
        )
        
        return temp_fahrenheit, temp_kelvin, seasonal_model
    
    def statistical_operations(self):
        """Comprehensive statistical analysis"""
        print("\n📊 Statistical operations...")
        
        temp_avg = self.temperature[:, :, 1]
        
        # Statistical measures along different axes
        daily_means = np.mean(temp_avg, axis=1)  # Average across stations each day
        station_means = np.mean(temp_avg, axis=0)  # Average across days each station
        overall_mean = np.mean(temp_avg)
        
        # Variability measures
        temp_std = np.std(temp_avg, axis=0)
        temp_var = np.var(temp_avg, axis=0)
        
        # Percentiles
        percentiles = np.percentile(temp_avg, [10, 25, 50, 75, 90], axis=0)
        
        # Correlation analysis
        temp_precip_corr = np.corrcoef(temp_avg.flatten(), self.precipitation.flatten())[0, 1]
        
        print(f"   Overall statistics:")
        print(f"     Mean temperature: {overall_mean:.2f}°C")
        print(f"     Temperature std range: {temp_std.min():.2f} to {temp_std.max():.2f}°C")
        print(f"     Temp-precipitation correlation: {temp_precip_corr:.3f}")
        
        return daily_means, station_means, percentiles, temp_std

# ============================================================================
# STEP 4: LINEAR ALGEBRA OPERATIONS
# ============================================================================

class LinearAlgebraAnalyzer:
    """Linear algebra operations for climate modeling"""
    
    def __init__(self, temperature: np.ndarray, precipitation: np.ndarray):
        self.temperature = temperature
        self.precipitation = precipitation
        
    def correlation_analysis(self):
        """Matrix operations for correlation analysis"""
        print("\n🔗 Correlation and covariance analysis...")
        
        # Correlation matrix between stations
        temp_matrix = self.temperature[:, :, 1].T  # Shape: (stations, days)
        temp_corr = np.corrcoef(temp_matrix)
        
        print(f"   Correlation matrix shape: {temp_corr.shape}")
        print(f"   Average inter-station correlation: {np.mean(temp_corr[temp_corr != 1]):.3f}")
        
        # Find highly correlated station pairs
        high_corr_mask = np.abs(temp_corr) > 0.8
        np.fill_diagonal(high_corr_mask, False)
        high_corr_pairs = np.where(high_corr_mask)
        print(f"   Highly correlated station pairs: {len(high_corr_pairs[0]) // 2}")
        
        return temp_corr, high_corr_pairs
    
    def principal_component_analysis(self):
        """PCA using eigenvalue decomposition"""
        print("\n🔍 Principal Component Analysis...")
        
        # Standardize data
        temp_data = self.temperature[:, :, 1]
        temp_standardized = (temp_data - np.mean(temp_data, axis=0)) / np.std(temp_data, axis=0)
        
        # Covariance matrix and eigendecomposition
        cov_matrix = np.cov(temp_standardized.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Explained variance
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Transform data
        principal_components = np.dot(temp_standardized, eigenvectors)
        
        print(f"   First PC explains {explained_variance_ratio[0]:.1%} of variance")
        print(f"   Components for 95% variance: {np.argmax(cumulative_variance > 0.95) + 1}")
        
        return eigenvalues, eigenvectors, principal_components, explained_variance_ratio
    
    def linear_regression_analysis(self):
        """Multiple linear regression using matrix operations"""
        print("\n📈 Linear regression analysis...")
        
        # Predict temperature using multiple factors
        days = np.arange(len(self.temperature))
        
        # Feature matrix
        X = np.column_stack([
            np.ones(len(days)),  # Intercept
            days,  # Linear trend
            np.sin(2 * np.pi * days / 365),  # Annual cycle
            np.cos(2 * np.pi * days / 365),  # Annual cycle
        ])
        
        # Target: average temperature across all stations
        y = np.mean(self.temperature[:, :, 1], axis=1)
        
        # Solve normal equations: β = (X^T X)^(-1) X^T y
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
        beta = np.linalg.solve(XtX, Xty)
        
        # Predictions and R-squared
        y_pred = np.dot(X, beta)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"   Linear regression R²: {r_squared:.4f}")
        print(f"   Annual trend: {beta[1] * 365:.3f} °C/year")
        
        return beta, y_pred, r_squared

# ============================================================================
# STEP 5: ADVANCED SCIENTIFIC ANALYSIS
# ============================================================================

class AdvancedAnalyzer:
    """Advanced scientific computing applications"""
    
    def __init__(self, temperature: np.ndarray, precipitation: np.ndarray):
        self.temperature = temperature
        self.precipitation = precipitation
        
    def fourier_analysis(self):
        """Frequency domain analysis of climate data"""
        print("\n🌀 Fourier analysis...")
        
        # Global temperature time series
        temp_series = np.mean(self.temperature[:, :, 1], axis=1)
        
        # Remove trend
        x = np.arange(len(temp_series))
        coeffs = np.polyfit(x, temp_series, 1)
        trend = np.polyval(coeffs, x)
        detrended = temp_series - trend
        
        # FFT analysis
        fft_result = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended), d=1)
        power = np.abs(fft_result)**2
        
        # Find dominant frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        if len(positive_freqs) > 1:
            periods = 1 / positive_freqs[1:]
            power_no_dc = positive_power[1:]
            
            if len(power_no_dc) > 0:
                peak_indices = np.where(power_no_dc > np.percentile(power_no_dc, 90))[0]
                dominant_periods = periods[peak_indices] if len(peak_indices) > 0 else []
            else:
                dominant_periods = []
        else:
            dominant_periods = []
        
        print(f"   Detected {len(dominant_periods)} dominant climate cycles")
        if len(dominant_periods) > 0:
            print(f"   Dominant periods (days): {dominant_periods[:3]}")
        
        return fft_result, freqs, dominant_periods, detrended
    
    def anomaly_detection(self):
        """Detect climate anomalies using statistical methods"""
        print("\n🚨 Anomaly detection...")
        
        temp_data = self.temperature[:, :, 1]
        
        # Z-score based anomaly detection
        temp_mean = np.mean(temp_data, axis=0)
        temp_std = np.std(temp_data, axis=0)
        
        # Avoid division by zero
        temp_std = np.where(temp_std == 0, 1, temp_std)
        z_scores = (temp_data - temp_mean) / temp_std
        
        # Define anomalies
        anomalies = np.abs(z_scores) > 3
        
        print(f"   Z-score anomalies: {anomalies.sum():,} events ({anomalies.sum()/anomalies.size*100:.2f}%)")
        
        return anomalies, z_scores

# ============================================================================
# STEP 6: DATA VISUALIZATION SYSTEM
# ============================================================================

class ClimateDataVisualizer:
    """Comprehensive visualization system for climate data"""
    
    def __init__(self, temperature: np.ndarray, precipitation: np.ndarray, 
                 years: int, stations: int):
        self.temperature = temperature
        self.precipitation = precipitation
        self.years = years
        self.stations = stations
        self.total_days = len(temperature)
        
        # Create date index
        self.dates = pd.date_range(start='2000-01-01', periods=self.total_days, freq='D')
        
        print(f"🎨 Climate Data Visualizer ready")
        print(f"   Will create comprehensive visualization suite")
    
    def create_comprehensive_visualization_suite(self):
        """Generate complete set of clean, readable climate visualizations"""
        print("\n🎨 GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        
        # Create separate, clean visualizations instead of cramming everything together
        self.create_main_climate_overview()
        self.create_statistical_analysis_plots()
        self.create_temporal_analysis_plots()
        self.create_advanced_visualizations()
        
        print("✅ All visualizations created successfully!")
    
    def create_main_climate_overview(self):
        """Create main climate overview with proper spacing"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Climate Data Overview', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Temperature time series (top left)
        self._plot_temperature_time_series(axes[0, 0])
        
        # 2. Seasonal patterns (top right)  
        self._plot_seasonal_patterns(axes[0, 1])
        
        # 3. Temperature distributions (bottom left)
        self._plot_temperature_distributions(axes[1, 0])
        
        # 4. Precipitation analysis (bottom right)
        self._plot_precipitation_analysis(axes[1, 1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('01_climate_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   📊 Created: 01_climate_overview.png")
    
    def create_statistical_analysis_plots(self):
        """Create statistical analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Station correlations (top left)
        self._plot_station_correlations(axes[0, 0])
        
        # 2. Climate trends (top right)
        self._plot_climate_trends(axes[0, 1])
        
        # 3. Extreme events (bottom left)
        self._plot_extreme_events(axes[1, 0])
        
        # 4. Monthly summary (bottom right)
        self._plot_monthly_summary(axes[1, 1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('02_statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   📈 Created: 02_statistical_analysis.png")
    
    def create_temporal_analysis_plots(self):
        """Create temporal analysis plots"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Temporal Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Detailed time series (top)
        self._plot_detailed_time_series(axes[0])
        
        # 2. Anomaly detection (bottom)
        self._plot_anomaly_detection(axes[1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('03_temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   ⏰ Created: 03_temporal_analysis.png")
    
    def _plot_detailed_time_series(self, ax):
        """Plot detailed time series with multiple components"""
        global_temp = np.mean(self.temperature[:, :, 1], axis=1)
        
        # Plot components
        ax.plot(self.dates, global_temp, alpha=0.6, color='lightblue', 
               linewidth=1, label='Daily Temperature')
        
        # Add moving averages
        if len(global_temp) > 30:
            window_30 = np.convolve(global_temp, np.ones(30)/30, mode='same')
            ax.plot(self.dates, window_30, color='blue', linewidth=2, label='30-day Average')
        
        if len(global_temp) > 365:
            window_365 = np.convolve(global_temp, np.ones(365)/365, mode='same')
            ax.plot(self.dates, window_365, color='red', linewidth=3, label='Annual Average')
        
        # Add linear trend
        days_numeric = np.arange(len(global_temp))
        trend_coeffs = np.polyfit(days_numeric, global_temp, 1)
        trend_line = np.polyval(trend_coeffs, days_numeric)
        ax.plot(self.dates, trend_line, '--', color='darkgreen', linewidth=2,
               label=f'Linear Trend: {trend_coeffs[0]*365:.4f}°C/year')
        
        ax.set_title('Detailed Temperature Time Series Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_temperature_time_series(self, ax):
        """Plot temperature time series with trends"""
        global_temp = np.mean(self.temperature[:, :, 1], axis=1)
        
        ax.plot(self.dates, global_temp, alpha=0.7, color='steelblue', linewidth=1)
        
        # Add moving average
        window = min(30, len(global_temp) // 10)
        if window > 1:
            moving_avg = np.convolve(global_temp, np.ones(window)/window, mode='same')
            ax.plot(self.dates, moving_avg, color='red', linewidth=2, label='Moving Average')
        
        ax.set_title('Global Temperature Time Series', fontweight='bold')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, alpha=0.3)
        if window > 1:
            ax.legend()
    
    def _plot_seasonal_patterns(self, ax):
        """Plot seasonal temperature patterns"""
        daily_temps = np.mean(self.temperature[:, :, 1], axis=1)
        years_complete = len(daily_temps) // 365
        
        if years_complete > 0:
            seasonal_data = daily_temps[:years_complete*365].reshape(years_complete, 365)
            seasonal_mean = np.mean(seasonal_data, axis=0)
            seasonal_std = np.std(seasonal_data, axis=0)
            
            days_of_year = np.arange(1, 366)
            ax.fill_between(days_of_year, 
                           seasonal_mean - seasonal_std, 
                           seasonal_mean + seasonal_std, 
                           alpha=0.3, color='lightblue')
            ax.plot(days_of_year, seasonal_mean, color='navy', linewidth=2)
        
        ax.set_title('Seasonal Temperature Patterns', fontweight='bold')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, alpha=0.3)
    
    def _plot_temperature_distributions(self, ax):
        """Plot temperature distribution analysis"""
        min_temps = self.temperature[:, :, 0].flatten()
        avg_temps = self.temperature[:, :, 1].flatten()
        max_temps = self.temperature[:, :, 2].flatten()
        
        ax.hist(min_temps, bins=30, alpha=0.6, label='Min', color='blue', density=True)
        ax.hist(avg_temps, bins=30, alpha=0.6, label='Avg', color='green', density=True)
        ax.hist(max_temps, bins=30, alpha=0.6, label='Max', color='red', density=True)
        
        ax.set_title('Temperature Distributions', fontweight='bold')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_precipitation_analysis(self, ax):
        """Plot precipitation patterns"""
        daily_precip = np.sum(self.precipitation, axis=1)
        
        dry_days = np.sum(daily_precip == 0)
        light_rain = np.sum((daily_precip > 0) & (daily_precip <= 10))
        moderate_rain = np.sum((daily_precip > 10) & (daily_precip <= 25))
        heavy_rain = np.sum(daily_precip > 25)
        
        categories = ['Dry\nDays', 'Light\nRain', 'Moderate\nRain', 'Heavy\nRain']
        counts = [dry_days, light_rain, moderate_rain, heavy_rain]
        colors = ['sandybrown', 'lightblue', 'royalblue', 'darkblue']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8)
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                   f'{count/total*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Precipitation Distribution', fontweight='bold')
        ax.set_ylabel('Number of Days')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_station_correlations(self, ax):
        """Plot correlation matrix heatmap"""
        # Sample subset for visibility
        n_stations_plot = min(self.stations, 15)
        if n_stations_plot > 1:
            station_indices = np.linspace(0, self.stations-1, n_stations_plot, dtype=int)
            temp_subset = self.temperature[:, station_indices, 1].T
            correlation_matrix = np.corrcoef(temp_subset)
            
            im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
            
            ax.set_title('Station Correlations', fontweight='bold')
            ax.set_xlabel('Station ID')
            ax.set_ylabel('Station ID')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'Need >1 station\nfor correlations', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Station Correlations', fontweight='bold')
    
    def _plot_climate_trends(self, ax):
        """Plot climate trends by station"""
        trends = []
        for station in range(self.stations):
            temp_series = self.temperature[:, station, 1]
            days = np.arange(len(temp_series))
            coeffs = np.polyfit(days, temp_series, 1)
            annual_trend = coeffs[0] * 365
            trends.append(annual_trend)
        
        station_ids = np.arange(1, self.stations + 1)
        colors = ['red' if trend > 0 else 'blue' for trend in trends]
        
        ax.bar(station_ids, trends, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_title('Climate Trends by Station', fontweight='bold')
        ax.set_xlabel('Station ID')
        ax.set_ylabel('Trend (°C/year)')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_extreme_events(self, ax):
        """Plot extreme events analysis"""
        temp_data = self.temperature[:, :, 1]
        hot_threshold = np.percentile(temp_data, 95)
        cold_threshold = np.percentile(temp_data, 5)
        
        # Count events by year
        years_complete = len(temp_data) // 365
        extreme_counts = []
        
        for year in range(min(years_complete, 5)):  # Max 5 years for visibility
            start_day = year * 365
            end_day = (year + 1) * 365
            year_temp = temp_data[start_day:end_day]
            
            hot_days = np.sum(year_temp > hot_threshold)
            cold_days = np.sum(year_temp < cold_threshold)
            extreme_counts.append([hot_days, cold_days])
        
        if extreme_counts:
            extreme_counts = np.array(extreme_counts)
            x = np.arange(len(extreme_counts))
            
            ax.bar(x, extreme_counts[:, 0], label='Hot Days', color='red', alpha=0.8)
            ax.bar(x, extreme_counts[:, 1], bottom=extreme_counts[:, 0],
                   label='Cold Days', color='blue', alpha=0.8)
            
            ax.set_xlabel('Year')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Y{i+1}' for i in range(len(extreme_counts))])
            ax.legend()
        
        ax.set_title('Extreme Events by Year', fontweight='bold')
        ax.set_ylabel('Number of Days')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_monthly_summary(self, ax):
        """Plot monthly climate summary"""
        days_per_month = 30
        months_available = min(12, len(self.temperature) // days_per_month)
        
        monthly_temp = []
        monthly_precip = []
        
        for month in range(months_available):
            start_day = month * days_per_month
            end_day = (month + 1) * days_per_month
            
            temp_avg = np.mean(self.temperature[start_day:end_day, :, 1])
            precip_total = np.mean(np.sum(self.precipitation[start_day:end_day], axis=0))
            
            monthly_temp.append(temp_avg)
            monthly_precip.append(precip_total)
        
        # Dual axis plot
        ax2 = ax.twinx()
        
        months_short = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        x = range(months_available)
        
        line1 = ax.plot(x, monthly_temp, 'o-', color='red', linewidth=2, label='Temperature')
        ax.set_ylabel('Temperature (°C)', color='red')
        
        bars = ax2.bar(x, monthly_precip, alpha=0.6, color='blue', label='Precipitation')
        ax2.set_ylabel('Precipitation (mm)', color='blue')
        
        ax.set_xlabel('Month')
        ax.set_title('Monthly Climate Summary', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(months_short[:months_available])
        ax.grid(True, alpha=0.3)
    
    def _plot_anomaly_detection(self, ax):
        """Plot anomaly detection results"""
        temp_data = self.temperature[:, :, 1]
        
        # Calculate z-scores for first station
        if self.stations > 0:
            station_data = temp_data[:, 0]
            mean_temp = np.mean(station_data)
            std_temp = np.std(station_data)
            
            if std_temp > 0:
                z_scores = (station_data - mean_temp) / std_temp
                anomalies = np.abs(z_scores) > 3
                
                ax.plot(self.dates, z_scores, alpha=0.7, color='blue', linewidth=1)
                ax.scatter(self.dates[anomalies], z_scores[anomalies], 
                          color='red', s=30, zorder=5, alpha=0.8)
                
                ax.axhline(y=3, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=-3, color='red', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No variation\nin data', 
                       ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Anomaly Detection (Station 1)', fontweight='bold')
        ax.set_ylabel('Z-Score')
        ax.grid(True, alpha=0.3)
    
    def create_advanced_visualizations(self):
        """Create clean advanced scientific visualizations"""
        
        # 1. Fourier Analysis
        self.create_fourier_analysis_plot()
        
        # 2. PCA Analysis
        self.create_pca_analysis_plot()
        
        # 3. Spatial Patterns
        self.create_spatial_analysis_plot()
    
    def create_fourier_analysis_plot(self):
        """Create clean Fourier analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        global_temp = np.mean(self.temperature[:, :, 1], axis=1)
        
        # Remove trend
        x = np.arange(len(global_temp))
        coeffs = np.polyfit(x, global_temp, 1)
        trend = np.polyval(coeffs, x)
        detrended = global_temp - trend
        
        # 1. Original vs detrended signal
        axes[0, 0].plot(self.dates, global_temp, label='Original Signal', alpha=0.8, color='blue')
        axes[0, 0].plot(self.dates, trend, '--', label='Linear Trend', color='red', linewidth=2)
        axes[0, 0].plot(self.dates, detrended + np.mean(global_temp), 
                       label='Detrended Signal', alpha=0.8, color='green')
        axes[0, 0].set_title('Signal Preprocessing')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Power spectral density
        fft_result = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended), d=1)
        power = np.abs(fft_result)**2
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        if len(positive_freqs) > 1:
            periods = 1 / positive_freqs[1:]
            power_periods = positive_power[1:]
            
            if len(periods) > 0:
                axes[0, 1].loglog(periods, power_periods, 'b-', alpha=0.8, linewidth=2)
                axes[0, 1].axvline(x=365, color='red', linestyle='--', linewidth=2, label='Annual')
                if max(periods) > 182:
                    axes[0, 1].axvline(x=365/2, color='orange', linestyle='--', linewidth=2, label='Semi-annual')
                axes[0, 1].legend()
        
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Period (days)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Dominant frequencies
        if len(positive_freqs) > 1 and len(periods) > 0:
            peak_threshold = np.percentile(power_periods, 85)
            peak_indices = np.where(power_periods > peak_threshold)[0]
            dominant_periods = periods[peak_indices][:8]  # Top 8
            dominant_power = power_periods[peak_indices][:8]
            
            if len(dominant_periods) > 0:
                bars = axes[1, 0].bar(range(len(dominant_periods)), dominant_power, 
                                     color='skyblue', edgecolor='black', alpha=0.8)
                
                for i, (bar, period) in enumerate(zip(bars, dominant_periods)):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                                   f'{period:.0f}d', ha='center', va='bottom', fontweight='bold')
        
        axes[1, 0].set_title('Dominant Periods')
        axes[1, 0].set_xlabel('Rank')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Signal reconstruction (show a portion)
        if len(detrended) > 100:
            sample_days = slice(0, min(730, len(detrended)))  # Show up to 2 years
            axes[1, 1].plot(self.dates[sample_days], detrended[sample_days], 
                           label='Original Detrended', alpha=0.8, color='blue')
            
            # Simple reconstruction using dominant frequency
            if len(positive_freqs) > 1:
                reconstructed = np.zeros_like(detrended)
                dominant_freq_idx = np.argmax(positive_power[1:]) + 1
                freq = positive_freqs[dominant_freq_idx]
                amplitude = np.abs(fft_result[dominant_freq_idx]) / len(detrended)
                phase = np.angle(fft_result[dominant_freq_idx])
                reconstructed = 2 * amplitude * np.cos(2 * np.pi * freq * x + phase)
                
                axes[1, 1].plot(self.dates[sample_days], reconstructed[sample_days], 
                               label='Reconstructed (Main freq)', linewidth=2, color='red')
        
        axes[1, 1].set_title('Signal Reconstruction')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Temperature Anomaly (°C)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('04_fourier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   🌀 Created: 04_fourier_analysis.png")
    
    def create_pca_analysis_plot(self):
        """Create clean PCA analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # Standardize temperature data
        temp_data = self.temperature[:, :, 1]
        temp_standardized = (temp_data - np.mean(temp_data, axis=0)) / np.std(temp_data, axis=0)
        
        # PCA calculation
        cov_matrix = np.cov(temp_standardized.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        # Calculate explained variance
        explained_variance = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance)
        
        # Transform data
        principal_components = np.dot(temp_standardized, eigenvectors)
        
        # 1. Scree plot
        n_components = min(15, len(explained_variance))
        axes[0, 0].plot(range(1, n_components + 1), explained_variance[:n_components], 
                       'bo-', markersize=8, linewidth=2)
        axes[0, 0].set_title('Scree Plot - Explained Variance')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add annotations for first few components
        for i in range(min(5, len(explained_variance))):
            axes[0, 0].annotate(f'{explained_variance[i]:.3f}', 
                               xy=(i+1, explained_variance[i]), 
                               xytext=(5, 5), textcoords='offset points')
        
        # 2. Cumulative explained variance
        axes[0, 1].plot(range(1, n_components + 1), cumulative_variance[:n_components], 
                       'ro-', markersize=8, linewidth=2)
        axes[0, 1].axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Variance')
        axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% Variance')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Principal components time series
        n_pcs_to_plot = min(3, principal_components.shape[1])
        for i in range(n_pcs_to_plot):
            axes[1, 0].plot(self.dates, principal_components[:, i], 
                           label=f'PC{i+1} ({explained_variance[i]:.1%})', alpha=0.8)
        
        axes[1, 0].set_title('Principal Components Time Series')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('PC Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Spatial patterns (eigenvectors)
        station_ids = np.arange(1, self.stations + 1)
        n_patterns = min(3, len(eigenvectors))
        for i in range(n_patterns):
            axes[1, 1].plot(station_ids, eigenvectors[:, i], 'o-', 
                           label=f'PC{i+1} Pattern', linewidth=2, markersize=6)
        
        axes[1, 1].set_title('Spatial Patterns (Eigenvectors)')
        axes[1, 1].set_xlabel('Weather Station ID')
        axes[1, 1].set_ylabel('Loading')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('05_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   🔍 Created: 05_pca_analysis.png")
        
        # Print summary
        print(f"      PCA Summary: PC1 explains {explained_variance[0]:.1%} of variance")
        if len(explained_variance) >= 3:
            print(f"      First 3 PCs explain {cumulative_variance[2]:.1%} of variance")
    
    def create_spatial_analysis_plot(self):
        """Create clean spatial analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spatial Climate Patterns', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Station mean temperatures
        station_means = np.mean(self.temperature[:, :, 1], axis=0)
        station_ids = np.arange(1, self.stations + 1)
        
        scatter = axes[0, 0].scatter(station_ids, station_means, c=station_means, 
                                   s=150, cmap='RdYlBu_r', edgecolors='black', linewidth=1)
        cbar = plt.colorbar(scatter, ax=axes[0, 0])
        cbar.set_label('Mean Temperature (°C)')
        
        axes[0, 0].set_title('Mean Temperature by Station')
        axes[0, 0].set_xlabel('Weather Station ID')
        axes[0, 0].set_ylabel('Mean Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Temperature variability
        station_stds = np.std(self.temperature[:, :, 1], axis=0)
        bars = axes[0, 1].bar(station_ids, station_stds, color='orange', alpha=0.7, edgecolor='black')
        
        # Highlight most and least variable
        max_var_idx = np.argmax(station_stds)
        min_var_idx = np.argmin(station_stds)
        bars[max_var_idx].set_color('red')
        bars[min_var_idx].set_color('blue')
        
        axes[0, 1].set_title('Temperature Variability by Station')
        axes[0, 1].set_xlabel('Weather Station ID')
        axes[0, 1].set_ylabel('Standard Deviation (°C)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Inter-station correlations (sample)
        if self.stations > 1:
            n_stations_plot = min(self.stations, 12)
            station_indices = np.linspace(0, self.stations-1, n_stations_plot, dtype=int)
            temp_subset = self.temperature[:, station_indices, 1].T
            correlation_matrix = np.corrcoef(temp_subset)
            
            im = axes[1, 0].imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
            cbar = plt.colorbar(im, ax=axes[1, 0])
            cbar.set_label('Correlation Coefficient')
            
            axes[1, 0].set_title('Inter-Station Correlations (Sample)')
            axes[1, 0].set_xlabel('Station Index')
            axes[1, 0].set_ylabel('Station Index')
        
        # 4. Climate zones (simple clustering)
        station_features = np.column_stack([
            station_means,
            station_stds,
            np.mean(self.precipitation, axis=0)
        ])
        
        # Simple clustering based on temperature characteristics
        cool_mask = station_means < np.percentile(station_means, 33)
        warm_mask = station_means > np.percentile(station_means, 67)
        moderate_mask = ~(cool_mask | warm_mask)
        
        axes[1, 1].scatter(station_ids[cool_mask], station_means[cool_mask], 
                          c='blue', s=100, alpha=0.8, label=f'Cool ({np.sum(cool_mask)})', 
                          edgecolors='black')
        axes[1, 1].scatter(station_ids[moderate_mask], station_means[moderate_mask], 
                          c='green', s=100, alpha=0.8, label=f'Moderate ({np.sum(moderate_mask)})', 
                          edgecolors='black')
        axes[1, 1].scatter(station_ids[warm_mask], station_means[warm_mask], 
                          c='red', s=100, alpha=0.8, label=f'Warm ({np.sum(warm_mask)})', 
                          edgecolors='black')
        
        axes[1, 1].set_title('Climate Zone Classification')
        axes[1, 1].set_xlabel('Weather Station ID')
        axes[1, 1].set_ylabel('Mean Temperature (°C)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('06_spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   🗺️ Created: 06_spatial_analysis.png")
    
    def _plot_fourier_analysis(self, ax):
        """Plot Fourier analysis results"""
        global_temp = np.mean(self.temperature[:, :, 1], axis=1)
        
        # Remove trend
        x = np.arange(len(global_temp))
        coeffs = np.polyfit(x, global_temp, 1)
        trend = np.polyval(coeffs, x)
        detrended = global_temp - trend
        
        # FFT
        fft_result = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended), d=1)
        power = np.abs(fft_result)**2
        
        # Plot power spectrum
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        if len(positive_freqs) > 1:
            periods = 1 / positive_freqs[1:]
            power_no_dc = positive_power[1:]
            
            if len(periods) > 0:
                ax.loglog(periods, power_no_dc, 'b-', alpha=0.8)
                ax.set_xlabel('Period (days)')
                ax.set_ylabel('Power')
                ax.grid(True, alpha=0.3)
        
        ax.set_title('Frequency Analysis', fontweight='bold')
    
    def _plot_pca_analysis(self, ax):
        """Plot PCA results"""
        temp_data = self.temperature[:, :, 1]
        temp_standardized = (temp_data - np.mean(temp_data, axis=0)) / np.std(temp_data, axis=0)
        
        # PCA calculation
        cov_matrix = np.cov(temp_standardized.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort and calculate explained variance
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        explained_variance = eigenvalues / np.sum(eigenvalues)
        
        # Plot scree plot
        n_components = min(10, len(explained_variance))
        ax.plot(range(1, n_components + 1), explained_variance[:n_components], 'bo-')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Scree Plot', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_spatial_patterns(self, ax):
        """Plot spatial patterns"""
        station_means = np.mean(self.temperature[:, :, 1], axis=0)
        station_ids = np.arange(1, self.stations + 1)
        
        scatter = ax.scatter(station_ids, [1]*self.stations, c=station_means, 
                           s=100, cmap='RdYlBu_r', edgecolors='black')
        
        plt.colorbar(scatter, ax=ax, label='Mean Temperature (°C)')
        ax.set_title('Spatial Temperature Patterns', fontweight='bold')
        ax.set_xlabel('Station ID')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_longterm_trends(self, ax):
        """Plot long-term climate trends"""
        global_temp = np.mean(self.temperature[:, :, 1], axis=1)
        
        # Annual averages if we have multiple years
        years_complete = len(global_temp) // 365
        if years_complete > 1:
            annual_temps = []
            for year in range(years_complete):
                start_day = year * 365
                end_day = (year + 1) * 365
                annual_avg = np.mean(global_temp[start_day:end_day])
                annual_temps.append(annual_avg)
            
            years = np.arange(1, years_complete + 1)
            ax.plot(years, annual_temps, 'bo-', linewidth=2, markersize=8)
            
            # Add trend line
            if len(annual_temps) > 1:
                trend_coeffs = np.polyfit(years, annual_temps, 1)
                trend_line = np.polyval(trend_coeffs, years)
                ax.plot(years, trend_line, 'r--', linewidth=2, 
                       label=f'Trend: {trend_coeffs[0]:.3f}°C/year')
                ax.legend()
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Annual Temperature (°C)')
        else:
            ax.text(0.5, 0.5, 'Need multiple years\nfor trend analysis', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Long-term Climate Trends', fontweight='bold')
        ax.grid(True, alpha=0.3)

# ============================================================================
# STEP 7: INTEGRATED ANALYSIS SYSTEM
# ============================================================================

class ClimateAnalysisSystem:
    """Complete integration of all NumPy techniques with visualizations"""
    
    def __init__(self, years: int = 5, stations: int = 20):
        print("="*80)
        print("🌍 COMPREHENSIVE NUMPY CLIMATE ANALYSIS SYSTEM")
        print("="*80)
        
        self.years = years
        self.stations = stations
        
        # Initialize data generator
        self.generator = ClimateDataGenerator(years, stations)
        
        # Generate climate data
        self.temperature = self.generator.generate_temperature_data()
        self.precipitation = self.generator.generate_precipitation_data()
        
        # Initialize analysis modules
        self.processor = DataProcessor(self.temperature, self.precipitation)
        self.math_analyzer = MathematicalAnalyzer(self.temperature, self.precipitation)
        self.linalg_analyzer = LinearAlgebraAnalyzer(self.temperature, self.precipitation)
        self.advanced_analyzer = AdvancedAnalyzer(self.temperature, self.precipitation)
        
        # Initialize visualizer
        self.visualizer = ClimateDataVisualizer(
            self.temperature, self.precipitation, years, stations
        )
        
        self.results = {}
        
    def run_comprehensive_analysis(self):
        """Execute complete analysis pipeline"""
        print("\n🔄 RUNNING COMPREHENSIVE ANALYSIS PIPELINE")
        start_time = time.time()
        
        # Phase 1: Data processing and indexing
        print("\n📋 Phase 1: Data Processing...")
        hot_days, extreme_weather = self.processor.demonstrate_indexing_techniques()
        yearly_data, temp_range = self.processor.reshape_and_transpose_operations()
        self.results['data_processing'] = {
            'hot_days': hot_days,
            'extreme_weather': extreme_weather,
            'yearly_data': yearly_data,
            'temp_range': temp_range
        }
        
        # Phase 2: Mathematical analysis
        print("\n🧮 Phase 2: Mathematical Analysis...")
        temp_f, temp_k, seasonal_model = self.math_analyzer.basic_mathematical_operations()
        daily_means, station_means, percentiles, temp_std = self.math_analyzer.statistical_operations()
        self.results['mathematical'] = {
            'temperature_conversions': (temp_f, temp_k),
            'statistics': (daily_means, station_means, percentiles, temp_std),
        }
        
        # Phase 3: Linear algebra
        print("\n🔢 Phase 3: Linear Algebra...")
        temp_corr, high_corr = self.linalg_analyzer.correlation_analysis()
        eigenvals, eigenvecs, pca_data, var_ratio = self.linalg_analyzer.principal_component_analysis()
        regression_coefs, predictions, r_squared = self.linalg_analyzer.linear_regression_analysis()
        self.results['linear_algebra'] = {
            'correlations': (temp_corr, high_corr),
            'pca': (eigenvals, eigenvecs, pca_data, var_ratio),
            'regression': (regression_coefs, predictions, r_squared),
        }
        
        # Phase 4: Advanced analysis
        print("\n🔬 Phase 4: Advanced Analysis...")
        fft_result, freqs, periods, detrended = self.advanced_analyzer.fourier_analysis()
        anomalies, z_scores = self.advanced_analyzer.anomaly_detection()
        self.results['advanced'] = {
            'fourier': (fft_result, freqs, periods, detrended),
            'anomalies': (anomalies, z_scores),
        }
        
        # Phase 5: Visualizations
        print("\n🎨 Phase 5: Creating Visualizations...")
        self.visualizer.create_comprehensive_visualization_suite()
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time:.2f} seconds")
        
        return self.results
    
    def generate_comprehensive_report(self, results: Dict):
        """Generate detailed analysis report"""
        print("\n" + "="*80)
        print("🌍 COMPREHENSIVE CLIMATE ANALYSIS REPORT")
        print("="*80)
        
        # Data summary
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Study period: {self.years} years ({self.temperature.shape[0]} days)")
        print(f"   Weather stations: {self.stations}")
        print(f"   Temperature range: {self.temperature.min():.1f}°C to {self.temperature.max():.1f}°C")
        print(f"   Total precipitation: {self.precipitation.sum():.0f}mm")
        
        # Statistical insights
        _, _, percentiles, temp_std = results['mathematical']['statistics']
        print(f"\n📈 STATISTICAL INSIGHTS:")
        print(f"   Temperature variability range: {temp_std.min():.2f} to {temp_std.max():.2f}°C")
        print(f"   Median temperature: {np.median(percentiles):.2f}°C")
        
        # Correlation analysis
        temp_corr, _ = results['linear_algebra']['correlations']
        if temp_corr.size > 1:
            avg_correlation = np.mean(temp_corr[temp_corr != 1])
            print(f"\n🔗 SPATIAL CORRELATIONS:")
            print(f"   Average inter-station correlation: {avg_correlation:.3f}")
        
        # PCA results
        _, _, _, var_ratio = results['linear_algebra']['pca']
        print(f"\n🔍 PRINCIPAL COMPONENT ANALYSIS:")
        print(f"   First component explains: {var_ratio[0]:.1%} of variance")
        if len(var_ratio) >= 3:
            print(f"   First 3 components explain: {var_ratio[:3].sum():.1%} of variance")
        
        # Trend analysis
        regression_coefs, _, r_squared = results['linear_algebra']['regression']
        print(f"\n📈 TEMPORAL TRENDS:")
        print(f"   Seasonal model R²: {r_squared:.4f}")
        print(f"   Linear trend: {regression_coefs[1] * 365:.3f} °C/year")
        
        # Frequency analysis
        _, _, periods, _ = results['advanced']['fourier']
        print(f"\n🌀 FREQUENCY ANALYSIS:")
        print(f"   Dominant cycles detected: {len(periods)}")
        if len(periods) > 0:
            annual_cycles = [p for p in periods if 300 < p < 400]
            if annual_cycles:
                print(f"   Annual cycle period: {annual_cycles[0]:.1f} days")
        
        # Anomaly detection
        anomalies, _ = results['advanced']['anomalies']
        print(f"\n🚨 ANOMALY DETECTION:")
        print(f"   Total anomalous events: {anomalies.sum()}")
        print(f"   Anomaly rate: {anomalies.sum() / anomalies.size * 100:.2f}%")
        
        print(f"\n🎨 VISUALIZATIONS CREATED:")
        print(f"   📊 01_climate_overview.png - Main climate data overview")
        print(f"   📈 02_statistical_analysis.png - Statistical analysis plots")
        print(f"   ⏰ 03_temporal_analysis.png - Time series and anomaly detection")
        print(f"   🌀 04_fourier_analysis.png - Frequency domain analysis")
        print(f"   🔍 05_pca_analysis.png - Principal component analysis")
        print(f"   🗺️ 06_spatial_analysis.png - Spatial patterns and climate zones")
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("All NumPy features successfully demonstrated with visualizations!")
        print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute the complete climate analysis project with visualizations
    """
    print("🚀 STARTING COMPREHENSIVE NUMPY + VISUALIZATION PROJECT")
    print("Research Question: How can we detect climate patterns and anomalies")
    print("using NumPy computational techniques and visual analysis?")
    print("-" * 80)
    
    # Create analysis system
    climate_system = ClimateAnalysisSystem(years=5, stations=20)
    
    # Run complete analysis
    results = climate_system.run_comprehensive_analysis()
    
    # Generate comprehensive report
    climate_system.generate_comprehensive_report(results)
    
    # Final summary
    print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("\nYou have demonstrated mastery of:")
    print("✅ NumPy array creation and manipulation")
    print("✅ Advanced indexing and slicing techniques")
    print("✅ Mathematical and statistical operations")
    print("✅ Linear algebra applications")
    print("✅ Performance optimization strategies")
    print("✅ Advanced scientific computing")
    print("✅ Professional data visualization")
    print("✅ Scientific storytelling through plots")
    print("✅ Real-world climate science applications")
    
    print("\n📁 FILES CREATED:")
    print("   📊 01_climate_overview.png - Temperature, seasonal patterns, distributions")
    print("   📈 02_statistical_analysis.png - Correlations, trends, extreme events") 
    print("   ⏰ 03_temporal_analysis.png - Detailed time series and anomalies")
    print("   🌀 04_fourier_analysis.png - Frequency analysis and signal processing")
    print("   🔍 05_pca_analysis.png - Principal component analysis results")
    print("   🗺️ 06_spatial_analysis.png - Spatial patterns and climate zones")
    
    print("\n🌟 Ready for real-world scientific computing challenges!")

if __name__ == "__main__":
    main()
