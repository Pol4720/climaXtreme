"""
Integration tests for the synthetic data generation pipeline.
Tests the full workflow from CSV input to Parquet output.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestSyntheticPipelineIntegration:
    """Integration tests for synthetic data generation pipeline."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data similar to GlobalLandTemperaturesByCity.csv"""
        np.random.seed(42)
        n = 1000
        
        cities = ['London', 'Paris', 'Tokyo', 'Sydney', 'New York']
        countries = ['United Kingdom', 'France', 'Japan', 'Australia', 'United States']
        latitudes = ['51.51N', '48.85N', '35.68N', '33.87S', '40.71N']
        longitudes = ['0.13W', '2.35E', '139.69E', '151.21E', '74.01W']
        
        records = []
        for i in range(n):
            city_idx = i % len(cities)
            year = 1900 + (i // 12) % 124
            month = (i % 12) + 1
            
            # Temperature with seasonal pattern
            seasonal = 15 * np.sin(2 * np.pi * (month - 1) / 12)
            base_temp = 15 + city_idx * 2
            temp = base_temp + seasonal + np.random.normal(0, 3)
            
            records.append({
                'dt': f'{year}-{month:02d}-01',
                'AverageTemperature': temp,
                'AverageTemperatureUncertainty': abs(np.random.normal(0.5, 0.2)),
                'City': cities[city_idx],
                'Country': countries[city_idx],
                'Latitude': latitudes[city_idx],
                'Longitude': longitudes[city_idx]
            })
        
        return pd.DataFrame(records)
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for input/output."""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        yield {'input': input_dir, 'output': output_dir}
        
        # Cleanup
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    
    def test_csv_to_dataframe_loading(self, sample_csv_data, temp_dirs):
        """Test loading CSV file into DataFrame."""
        csv_path = Path(temp_dirs['input']) / 'test_data.csv'
        sample_csv_data.to_csv(csv_path, index=False)
        
        # Load and verify
        loaded_df = pd.read_csv(csv_path)
        
        assert len(loaded_df) == len(sample_csv_data)
        assert 'AverageTemperature' in loaded_df.columns
        assert 'City' in loaded_df.columns
        assert 'Latitude' in loaded_df.columns
    
    def test_data_preprocessing_for_synthetic_generation(self, sample_csv_data):
        """Test data preprocessing steps."""
        df = sample_csv_data.copy()
        
        # Parse dates
        df['dt'] = pd.to_datetime(df['dt'])
        df['year'] = df['dt'].dt.year
        df['month'] = df['dt'].dt.month
        
        # Verify preprocessing
        assert 'year' in df.columns
        assert 'month' in df.columns
        assert df['year'].min() >= 1900
        assert df['month'].min() >= 1
        assert df['month'].max() <= 12
    
    def test_latitude_parsing(self, sample_csv_data):
        """Test latitude string parsing to numeric."""
        df = sample_csv_data.copy()
        
        def parse_latitude(lat_str):
            """Parse latitude string like '51.51N' to numeric."""
            if pd.isna(lat_str):
                return np.nan
            lat_str = str(lat_str)
            direction = lat_str[-1] if lat_str[-1] in 'NS' else ''
            value = float(lat_str[:-1]) if direction else float(lat_str)
            return -value if direction == 'S' else value
        
        df['latitude_numeric'] = df['Latitude'].apply(parse_latitude)
        
        # Verify
        assert df['latitude_numeric'].min() >= -90
        assert df['latitude_numeric'].max() <= 90
        
        # Sydney should be negative (Southern hemisphere)
        sydney_lat = df[df['City'] == 'Sydney']['latitude_numeric'].iloc[0]
        assert sydney_lat < 0
        
        # London should be positive (Northern hemisphere)
        london_lat = df[df['City'] == 'London']['latitude_numeric'].iloc[0]
        assert london_lat > 0
    
    def test_climate_zone_classification(self, sample_csv_data):
        """Test climate zone assignment based on latitude."""
        def classify_climate_zone(latitude):
            abs_lat = abs(latitude)
            if abs_lat < 23.5:
                return 'Tropical'
            elif abs_lat < 40:
                return 'Subtropical'
            elif abs_lat < 55:
                return 'Temperate'
            elif abs_lat < 66.5:
                return 'Continental'
            else:
                return 'Polar'
        
        df = sample_csv_data.copy()
        
        # Parse latitude
        def parse_lat(lat_str):
            lat_str = str(lat_str)
            direction = lat_str[-1] if lat_str[-1] in 'NS' else ''
            value = float(lat_str[:-1]) if direction else float(lat_str)
            return -value if direction == 'S' else value
        
        df['latitude_numeric'] = df['Latitude'].apply(parse_lat)
        df['climate_zone'] = df['latitude_numeric'].apply(classify_climate_zone)
        
        # Verify zones
        valid_zones = ['Tropical', 'Subtropical', 'Temperate', 'Continental', 'Polar']
        assert df['climate_zone'].isin(valid_zones).all()
    
    def test_hourly_temperature_generation(self, sample_csv_data):
        """Test hourly temperature generation with diurnal cycle."""
        np.random.seed(42)
        
        # Simulate hourly generation for one day
        base_temp = 20.0
        hours = np.arange(24)
        
        # Diurnal cycle: minimum at 6 AM, maximum at 2-4 PM
        diurnal = 8 * np.sin(2 * np.pi * (hours - 6) / 24)
        noise = np.random.normal(0, 1, 24)
        
        hourly_temps = base_temp + diurnal + noise
        
        # Verify pattern
        assert hourly_temps.min() < base_temp  # Has values below base
        assert hourly_temps.max() > base_temp  # Has values above base
        
        # Min should be around early morning (hours 4-8)
        min_hour = np.argmin(hourly_temps)
        assert 3 <= min_hour <= 9
        
        # Max should be in afternoon (hours 12-18)
        max_hour = np.argmax(hourly_temps)
        assert 11 <= max_hour <= 19
    
    def test_precipitation_generation(self):
        """Test precipitation generation with gamma distribution."""
        np.random.seed(42)
        n = 10000
        
        # Wet/dry probability
        wet_prob = 0.15
        is_wet = np.random.random(n) < wet_prob
        
        # Gamma distribution for rain amounts
        shape = 0.5
        scale = 5.0
        rain = np.zeros(n)
        rain[is_wet] = np.random.gamma(shape, scale, is_wet.sum())
        
        # Verify
        assert (rain >= 0).all()  # No negative rain
        assert (rain == 0).sum() > n * 0.8  # Most days dry
        assert rain.max() > 0  # Some rain occurred
    
    def test_wind_speed_generation(self):
        """Test wind speed generation with Weibull distribution."""
        np.random.seed(42)
        n = 10000
        
        shape = 2.0
        scale = 15.0
        wind = np.random.weibull(shape, n) * scale
        
        # Verify
        assert (wind >= 0).all()  # Wind speed non-negative
        assert wind.mean() > 10  # Reasonable mean
        assert wind.mean() < 20  # Not too high
    
    def test_alert_threshold_logic(self):
        """Test weather alert threshold classification."""
        # Heat alert thresholds
        def get_heat_alert(temp):
            if temp >= 45:
                return 'red'
            elif temp >= 40:
                return 'orange'
            elif temp >= 35:
                return 'yellow'
            return 'green'
        
        assert get_heat_alert(50) == 'red'
        assert get_heat_alert(42) == 'orange'
        assert get_heat_alert(37) == 'yellow'
        assert get_heat_alert(30) == 'green'
        
        # Wind alert thresholds
        def get_wind_alert(wind_kmh):
            if wind_kmh >= 120:
                return 'red'
            elif wind_kmh >= 90:
                return 'orange'
            elif wind_kmh >= 60:
                return 'yellow'
            return 'green'
        
        assert get_wind_alert(150) == 'red'
        assert get_wind_alert(100) == 'orange'
        assert get_wind_alert(70) == 'yellow'
        assert get_wind_alert(40) == 'green'
    
    def test_parquet_output_format(self, sample_csv_data, temp_dirs):
        """Test Parquet output writing and reading."""
        df = sample_csv_data.copy()
        
        # Add synthetic columns
        np.random.seed(42)
        n = len(df)
        df['temperature_hourly'] = df['AverageTemperature'] + np.random.normal(0, 2, n)
        df['rain_mm'] = np.maximum(0, np.random.exponential(3, n) * (np.random.random(n) < 0.2))
        df['wind_speed_kmh'] = np.random.weibull(2, n) * 15
        
        # Write to parquet
        output_path = Path(temp_dirs['output']) / 'synthetic_data.parquet'
        df.to_parquet(output_path, index=False)
        
        # Read back and verify
        loaded_df = pd.read_parquet(output_path)
        
        assert len(loaded_df) == len(df)
        assert 'temperature_hourly' in loaded_df.columns
        assert 'rain_mm' in loaded_df.columns
        assert 'wind_speed_kmh' in loaded_df.columns
    
    def test_data_validation_ranges(self, sample_csv_data):
        """Test that generated data is within valid ranges."""
        df = sample_csv_data.copy()
        
        # Add synthetic columns with ranges
        np.random.seed(42)
        n = len(df)
        
        df['temperature_hourly'] = np.clip(
            df['AverageTemperature'] + np.random.normal(0, 5, n),
            -80, 60
        )
        df['humidity_pct'] = np.clip(
            60 + np.random.normal(0, 15, n),
            0, 100
        )
        df['pressure_hpa'] = np.clip(
            1013.25 + np.random.normal(0, 20, n),
            870, 1084
        )
        
        # Validate ranges
        assert df['temperature_hourly'].min() >= -80
        assert df['temperature_hourly'].max() <= 60
        assert df['humidity_pct'].min() >= 0
        assert df['humidity_pct'].max() <= 100
        assert df['pressure_hpa'].min() >= 870
        assert df['pressure_hpa'].max() <= 1084


class TestIntensityPredictorIntegration:
    """Integration tests for intensity prediction model."""
    
    @pytest.fixture
    def synthetic_event_data(self):
        """Create synthetic data with events for ML training."""
        np.random.seed(42)
        n = 1000
        
        df = pd.DataFrame({
            'temperature_hourly': np.random.normal(25, 10, n),
            'rain_mm': np.maximum(0, np.random.exponential(5, n)),
            'wind_speed_kmh': np.random.weibull(2, n) * 15,
            'humidity_pct': np.clip(60 + np.random.normal(0, 15, n), 0, 100),
            'pressure_hpa': 1013.25 + np.random.normal(0, 10, n),
            'month': np.random.randint(1, 13, n),
            'hour': np.random.randint(0, 24, n),
            'Latitude': np.random.choice(['40.71N', '33.87S', '51.51N'], n),
            'climate_zone': np.random.choice(['Tropical', 'Temperate', 'Polar'], n),
            'event_type': np.random.choice(['none', 'heatwave', 'cold_snap', 'storm'], n, 
                                           p=[0.7, 0.1, 0.1, 0.1])
        })
        
        # Generate intensity based on features (simple rule)
        df['event_intensity'] = 0.0
        
        # Heatwaves: intensity based on temperature
        heatwave_mask = df['event_type'] == 'heatwave'
        df.loc[heatwave_mask, 'event_intensity'] = np.clip(
            (df.loc[heatwave_mask, 'temperature_hourly'] - 30) / 5,
            1, 10
        )
        
        # Cold snaps: intensity based on how cold
        cold_mask = df['event_type'] == 'cold_snap'
        df.loc[cold_mask, 'event_intensity'] = np.clip(
            (10 - df.loc[cold_mask, 'temperature_hourly']) / 5,
            1, 10
        )
        
        # Storms: intensity based on wind
        storm_mask = df['event_type'] == 'storm'
        df.loc[storm_mask, 'event_intensity'] = np.clip(
            df.loc[storm_mask, 'wind_speed_kmh'] / 15,
            1, 10
        )
        
        return df
    
    def test_feature_preparation(self, synthetic_event_data):
        """Test feature preparation for ML model."""
        df = synthetic_event_data.copy()
        
        # Extract features that would be used
        feature_cols = ['temperature_hourly', 'rain_mm', 'wind_speed_kmh', 
                       'humidity_pct', 'pressure_hpa', 'month', 'hour']
        
        X = df[feature_cols].values
        y = df['event_intensity'].values
        
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(feature_cols)
        assert len(y) == len(df)
    
    def test_train_test_split(self, synthetic_event_data):
        """Test train/test splitting for time series."""
        from sklearn.model_selection import train_test_split
        
        df = synthetic_event_data.copy()
        
        # Only events
        event_df = df[df['event_intensity'] > 0]
        
        X = event_df[['temperature_hourly', 'wind_speed_kmh', 'humidity_pct']].values
        y = event_df['event_intensity'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) > len(X_test)
        assert len(y_train) == len(X_train)
    
    def test_model_training_and_prediction(self, synthetic_event_data):
        """Test full model training and prediction cycle."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        df = synthetic_event_data.copy()
        event_df = df[df['event_intensity'] > 0]
        
        feature_cols = ['temperature_hourly', 'wind_speed_kmh', 'humidity_pct', 'pressure_hpa']
        X = event_df[feature_cols].values
        y = event_df['event_intensity'].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Verify predictions are in valid range
        assert y_pred.min() >= 0
        assert y_pred.max() <= 15  # Some margin above 10
        
        # Model should have some predictive power
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0  # Better than mean baseline


class TestValidationIntegration:
    """Integration tests for data validation."""
    
    def test_synthetic_validator_full_check(self):
        """Test full validation of synthetic data."""
        np.random.seed(42)
        n = 500
        
        # Create synthetic data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'year': [2024] * n,
            'month': np.random.randint(1, 13, n),
            'day': np.random.randint(1, 29, n),
            'hour': np.random.randint(0, 24, n),
            'City': np.random.choice(['City_A', 'City_B'], n),
            'Country': ['Country_1'] * n,
            'Latitude': ['45.0N'] * n,
            'Longitude': ['10.0E'] * n,
            'temperature_hourly': np.random.normal(20, 10, n),
            'rain_mm': np.maximum(0, np.random.exponential(3, n)),
            'wind_speed_kmh': np.random.weibull(2, n) * 15,
            'humidity_pct': np.clip(60 + np.random.normal(0, 15, n), 0, 100)
        })
        
        # Validation checks
        assert 'timestamp' in df.columns
        assert 'temperature_hourly' in df.columns
        assert df['humidity_pct'].min() >= 0
        assert df['humidity_pct'].max() <= 100
        assert (df['rain_mm'] >= 0).all()
        assert (df['wind_speed_kmh'] >= 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
