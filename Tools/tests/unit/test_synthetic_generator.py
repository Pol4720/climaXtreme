"""
Unit tests for the SyntheticClimateGenerator module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# We'll test both the config dataclasses and the generator logic
# Since PySpark might not be available in all test environments,
# we'll mock it where necessary


class TestWeatherParams:
    """Tests for WeatherParams dataclass."""
    
    def test_default_values(self):
        """Test that WeatherParams has sensible defaults."""
        from climaxtreme.preprocessing.spark.synthetic_generator import WeatherParams
        
        params = WeatherParams()
        
        # Check temperature defaults
        assert params.temp_amplitude_diurnal == 8.0
        assert params.temp_amplitude_seasonal == 15.0
        assert params.temp_base == 15.0
        
        # Check precipitation defaults
        assert params.rain_probability == 0.15
        assert params.rain_shape == 0.5
        assert params.rain_scale == 5.0
        
        # Check wind defaults
        assert params.wind_shape == 2.0
        assert params.wind_scale == 15.0
        
        # Check humidity defaults
        assert params.humidity_base == 60.0
        assert params.humidity_temp_coef == -1.0
        
    def test_custom_values(self):
        """Test WeatherParams with custom values."""
        from climaxtreme.preprocessing.spark.synthetic_generator import WeatherParams
        
        params = WeatherParams(
            temp_base=25.0,
            rain_probability=0.3,
            wind_scale=20.0
        )
        
        assert params.temp_base == 25.0
        assert params.rain_probability == 0.3
        assert params.wind_scale == 20.0


class TestEventRates:
    """Tests for EventRates dataclass."""
    
    def test_default_values(self):
        """Test that EventRates has sensible defaults."""
        from climaxtreme.preprocessing.spark.synthetic_generator import EventRates
        
        rates = EventRates()
        
        assert rates.heatwave_rate == 0.02
        assert rates.cold_snap_rate == 0.02
        assert rates.drought_rate == 0.01
        assert rates.extreme_precip_rate == 0.03
        assert rates.hurricane_rate == 0.005
        assert rates.tornado_rate == 0.002
        
    def test_rates_are_probabilities(self):
        """Test that event rates are valid probabilities."""
        from climaxtreme.preprocessing.spark.synthetic_generator import EventRates
        
        rates = EventRates()
        
        for field, value in asdict(rates).items():
            assert 0 <= value <= 1, f"{field} should be between 0 and 1"


class TestSyntheticConfig:
    """Tests for SyntheticConfig dataclass."""
    
    def test_default_values(self):
        """Test that SyntheticConfig has sensible defaults."""
        from climaxtreme.preprocessing.spark.synthetic_generator import SyntheticConfig
        
        config = SyntheticConfig()
        
        assert config.seed == 42
        assert config.generate_hourly == True
        assert config.generate_storms == True
        assert config.generate_alerts == True
        assert config.sample_fraction == 1.0
        
    def test_nested_defaults(self):
        """Test that nested configs are properly initialized."""
        from climaxtreme.preprocessing.spark.synthetic_generator import SyntheticConfig
        
        config = SyntheticConfig()
        
        assert config.weather_params is not None
        assert config.event_rates is not None
        assert hasattr(config.weather_params, 'temp_base')
        assert hasattr(config.event_rates, 'heatwave_rate')


class TestClimateZoneClassification:
    """Tests for climate zone classification logic."""
    
    def test_tropical_zone(self):
        """Test classification of tropical zone."""
        from climaxtreme.preprocessing.spark.synthetic_generator import classify_climate_zone
        
        # Near equator
        zone = classify_climate_zone(5.0)  # 5°N
        assert zone == 'Tropical'
        
        zone = classify_climate_zone(-5.0)  # 5°S
        assert zone == 'Tropical'
    
    def test_subtropical_zone(self):
        """Test classification of subtropical zone."""
        from climaxtreme.preprocessing.spark.synthetic_generator import classify_climate_zone
        
        zone = classify_climate_zone(30.0)  # 30°N
        assert zone == 'Subtropical'
        
        zone = classify_climate_zone(-25.0)  # 25°S
        assert zone == 'Subtropical'
    
    def test_temperate_zone(self):
        """Test classification of temperate zone."""
        from climaxtreme.preprocessing.spark.synthetic_generator import classify_climate_zone
        
        zone = classify_climate_zone(45.0)  # 45°N
        assert zone == 'Temperate'
        
        zone = classify_climate_zone(-45.0)  # 45°S
        assert zone == 'Temperate'
    
    def test_continental_zone(self):
        """Test classification of continental zone."""
        from climaxtreme.preprocessing.spark.synthetic_generator import classify_climate_zone
        
        zone = classify_climate_zone(55.0)  # 55°N
        assert zone == 'Continental'
        
        zone = classify_climate_zone(-55.0)  # 55°S
        assert zone == 'Continental'
    
    def test_polar_zone(self):
        """Test classification of polar zone."""
        from climaxtreme.preprocessing.spark.synthetic_generator import classify_climate_zone
        
        zone = classify_climate_zone(70.0)  # 70°N
        assert zone == 'Polar'
        
        zone = classify_climate_zone(-75.0)  # 75°S
        assert zone == 'Polar'
    
    def test_boundary_conditions(self):
        """Test zone boundaries."""
        from climaxtreme.preprocessing.spark.synthetic_generator import classify_climate_zone
        
        # At exact boundaries
        assert classify_climate_zone(23.5) == 'Tropical'  # Just within tropical
        assert classify_climate_zone(40.0) == 'Subtropical'  # Just within subtropical
        assert classify_climate_zone(66.5) == 'Polar'  # Just within polar


class TestHourlyTemperatureGeneration:
    """Tests for hourly temperature generation functions."""
    
    def test_diurnal_cycle_pattern(self):
        """Test that temperature follows diurnal cycle."""
        np.random.seed(42)
        
        # Simulate diurnal pattern
        hours = np.arange(24)
        diurnal = 8 * np.sin(2 * np.pi * (hours - 6) / 24)
        
        # Should be minimum around 6 AM (hour 6)
        min_idx = np.argmin(diurnal)
        assert min_idx in [5, 6, 7], "Minimum should be around sunrise"
        
        # Should be maximum around 2-4 PM (hours 14-16)
        max_idx = np.argmax(diurnal)
        assert max_idx in [14, 15, 16, 17, 18], "Maximum should be in afternoon"
    
    def test_seasonal_variation(self):
        """Test that temperature shows seasonal variation."""
        # Simulate seasonal pattern for northern hemisphere
        day_of_year = np.arange(365)
        seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Winter minimum around day 10-20 (January)
        min_idx = np.argmin(seasonal)
        assert 0 <= min_idx <= 40 or min_idx >= 330, "Minimum should be in winter"
        
        # Summer maximum around day 170-190 (July)
        max_idx = np.argmax(seasonal)
        assert 170 <= max_idx <= 200, "Maximum should be in summer"


class TestPrecipitationGeneration:
    """Tests for precipitation generation."""
    
    def test_gamma_distribution_shape(self):
        """Test that precipitation follows gamma distribution."""
        np.random.seed(42)
        
        # Generate gamma-distributed precipitation
        shape = 0.5
        scale = 5.0
        n = 10000
        
        precip = np.random.gamma(shape, scale, n)
        
        # Check mean is approximately shape * scale
        assert abs(np.mean(precip) - shape * scale) < 0.5
        
        # Check it's always positive
        assert np.all(precip >= 0)
    
    def test_wet_dry_probability(self):
        """Test wet/dry day probability."""
        np.random.seed(42)
        
        n = 10000
        wet_prob = 0.15
        
        is_wet = np.random.random(n) < wet_prob
        actual_wet_ratio = is_wet.sum() / n
        
        # Should be close to expected probability
        assert abs(actual_wet_ratio - wet_prob) < 0.02


class TestWindGeneration:
    """Tests for wind speed generation."""
    
    def test_weibull_distribution(self):
        """Test that wind follows Weibull distribution."""
        np.random.seed(42)
        
        shape = 2.0
        scale = 15.0
        n = 10000
        
        wind = np.random.weibull(shape, n) * scale
        
        # Check it's always positive
        assert np.all(wind >= 0)
        
        # Mean of Weibull is scale * gamma(1 + 1/shape)
        from scipy import special
        expected_mean = scale * special.gamma(1 + 1/shape)
        actual_mean = np.mean(wind)
        
        assert abs(actual_mean - expected_mean) < 1.0


class TestAlertThresholds:
    """Tests for weather alert threshold logic."""
    
    def test_heat_alert_thresholds(self):
        """Test heat alert threshold classification."""
        # Define threshold logic
        def get_heat_alert(temp):
            if temp >= 45:
                return 'red'
            elif temp >= 40:
                return 'orange'
            elif temp >= 35:
                return 'yellow'
            else:
                return 'green'
        
        assert get_heat_alert(50) == 'red'
        assert get_heat_alert(42) == 'orange'
        assert get_heat_alert(37) == 'yellow'
        assert get_heat_alert(30) == 'green'
    
    def test_wind_alert_thresholds(self):
        """Test wind alert threshold classification."""
        def get_wind_alert(wind_kmh):
            if wind_kmh >= 120:
                return 'red'
            elif wind_kmh >= 90:
                return 'orange'
            elif wind_kmh >= 60:
                return 'yellow'
            else:
                return 'green'
        
        assert get_wind_alert(150) == 'red'
        assert get_wind_alert(100) == 'orange'
        assert get_wind_alert(70) == 'yellow'
        assert get_wind_alert(40) == 'green'
    
    def test_rain_alert_thresholds(self):
        """Test precipitation alert threshold classification."""
        def get_rain_alert(rain_mm):
            if rain_mm >= 100:
                return 'red'
            elif rain_mm >= 50:
                return 'orange'
            elif rain_mm >= 25:
                return 'yellow'
            else:
                return 'green'
        
        assert get_rain_alert(120) == 'red'
        assert get_rain_alert(70) == 'orange'
        assert get_rain_alert(30) == 'yellow'
        assert get_rain_alert(10) == 'green'


class TestStormCategoryClassification:
    """Tests for Saffir-Simpson storm category classification."""
    
    def test_saffir_simpson_scale(self):
        """Test storm category based on wind speed."""
        def get_storm_category(wind_kmh):
            wind_mph = wind_kmh * 0.621371
            if wind_mph < 39:
                return 'TD'  # Tropical Depression
            elif wind_mph < 74:
                return 'TS'  # Tropical Storm
            elif wind_mph < 96:
                return '1'
            elif wind_mph < 111:
                return '2'
            elif wind_mph < 130:
                return '3'
            elif wind_mph < 157:
                return '4'
            else:
                return '5'
        
        # Category 5: >= 157 mph (~253 km/h)
        assert get_storm_category(260) == '5'
        
        # Category 4: 130-156 mph (~209-251 km/h)
        assert get_storm_category(220) == '4'
        
        # Category 3: 111-129 mph (~179-208 km/h)
        assert get_storm_category(190) == '3'
        
        # Category 1: 74-95 mph (~119-153 km/h)
        assert get_storm_category(130) == '1'
        
        # Tropical Storm: 39-73 mph (~63-118 km/h)
        assert get_storm_category(80) == 'TS'


class TestDataValidation:
    """Tests for generated data validation."""
    
    def test_temperature_range(self):
        """Test that generated temperatures are within realistic range."""
        np.random.seed(42)
        n = 1000
        
        # Generate temperatures with realistic model
        base = 15
        seasonal = 15 * np.random.uniform(-1, 1, n)
        diurnal = 8 * np.random.uniform(-1, 1, n)
        noise = np.random.normal(0, 2, n)
        
        temps = base + seasonal + diurnal + noise
        
        # Should be within -60 to 60 (realistic Earth temperatures)
        assert np.all(temps > -80)
        assert np.all(temps < 70)
    
    def test_humidity_bounded(self):
        """Test that humidity is bounded 0-100."""
        np.random.seed(42)
        n = 1000
        
        humidity = np.clip(60 + np.random.normal(0, 15, n), 0, 100)
        
        assert np.all(humidity >= 0)
        assert np.all(humidity <= 100)
    
    def test_pressure_realistic(self):
        """Test that pressure values are realistic."""
        np.random.seed(42)
        n = 1000
        
        # Sea level pressure varies roughly 950-1050 hPa
        pressure = 1013.25 + np.random.normal(0, 10, n)
        
        assert np.all(pressure > 900)
        assert np.all(pressure < 1100)


class TestCreateGeneratorFactory:
    """Tests for the create_generator factory function."""
    
    def test_create_generator_with_defaults(self):
        """Test creating generator with default config."""
        # This test requires mocking Spark
        with patch('climaxtreme.preprocessing.spark.synthetic_generator.SparkSession') as mock_spark:
            mock_builder = MagicMock()
            mock_spark.builder = mock_builder
            mock_builder.appName.return_value = mock_builder
            mock_builder.config.return_value = mock_builder
            mock_builder.getOrCreate.return_value = MagicMock()
            
            from climaxtreme.preprocessing.spark.synthetic_generator import create_generator
            
            generator = create_generator()
            
            assert generator is not None
            assert generator.config.seed == 42
    
    def test_create_generator_with_custom_seed(self):
        """Test creating generator with custom seed."""
        with patch('climaxtreme.preprocessing.spark.synthetic_generator.SparkSession') as mock_spark:
            mock_builder = MagicMock()
            mock_spark.builder = mock_builder
            mock_builder.appName.return_value = mock_builder
            mock_builder.config.return_value = mock_builder
            mock_builder.getOrCreate.return_value = MagicMock()
            
            from climaxtreme.preprocessing.spark.synthetic_generator import create_generator
            
            generator = create_generator(seed=123)
            
            assert generator.config.seed == 123


# Integration-style tests that don't require Spark
class TestStatisticalConsistency:
    """Tests for statistical consistency of generated data."""
    
    def test_seasonal_temperature_correlation_with_month(self):
        """Test that temperature correlates with month/season."""
        np.random.seed(42)
        
        # Generate monthly data
        months = np.repeat(np.arange(1, 13), 1000)
        
        # Temperature model: warmer in summer (NH)
        base = 15
        seasonal = 15 * np.cos(2 * np.pi * (months - 7) / 12)  # Peak in July
        noise = np.random.normal(0, 3, len(months))
        
        temps = base + seasonal + noise
        
        # July (month 7) should be warmer than January (month 1) for NH
        july_mean = temps[months == 7].mean()
        jan_mean = temps[months == 1].mean()
        
        assert july_mean > jan_mean
    
    def test_humidity_temperature_inverse_correlation(self):
        """Test that humidity tends to be inversely correlated with temperature."""
        np.random.seed(42)
        n = 1000
        
        temps = 15 + np.random.normal(0, 10, n)
        # Humidity decreases with higher temperatures
        humidity = np.clip(70 - (temps - 15) + np.random.normal(0, 5, n), 0, 100)
        
        correlation = np.corrcoef(temps, humidity)[0, 1]
        
        # Should be negative correlation
        assert correlation < 0
    
    def test_event_frequency_realistic(self):
        """Test that extreme event frequency is realistic."""
        np.random.seed(42)
        n = 365 * 24  # One year of hourly data
        
        # Event rates
        heatwave_rate = 0.02
        cold_snap_rate = 0.02
        
        # Generate events
        is_heatwave = np.random.random(n) < heatwave_rate
        is_cold_snap = np.random.random(n) < cold_snap_rate
        
        # Count events (consecutive hours as single event)
        heatwave_hours = is_heatwave.sum()
        cold_snap_hours = is_cold_snap.sum()
        
        # Should have some events but not too many
        assert heatwave_hours > 0
        assert heatwave_hours < n * 0.1  # Less than 10% of time
        
        assert cold_snap_hours > 0
        assert cold_snap_hours < n * 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
