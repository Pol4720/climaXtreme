"""
Climate predictor with ensemble methods and advanced features.
Includes intensity prediction for extreme weather events.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from .baseline import BaselineModel


logger = logging.getLogger(__name__)


class ClimatePredictor:
    """
    Advanced climate predictor using ensemble methods.
    
    Combines multiple baseline models for improved prediction accuracy
    and provides uncertainty quantification.
    """
    
    def __init__(self, models: Optional[List[str]] = None) -> None:
        """
        Initialize climate predictor.
        
        Args:
            models: List of model types to include in ensemble
        """
        if models is None:
            models = ['linear', 'ridge', 'random_forest']
        
        self.model_types = models
        self.base_models: Dict[str, BaselineModel] = {}
        self.ensemble_model: Optional[VotingRegressor] = None
        self.is_fitted = False
        
        # Initialize base models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all base models."""
        for model_type in self.model_types:
            self.base_models[model_type] = BaselineModel(model_type)
        
        logger.info(f"Initialized ensemble with models: {self.model_types}")
    
    def train_ensemble(
        self, 
        df: pd.DataFrame,
        use_time_series_split: bool = True,
        n_splits: int = 5
    ) -> Dict[str, any]:
        """
        Train ensemble of models with time series cross-validation.
        
        Args:
            df: Training data DataFrame
            use_time_series_split: Whether to use time series split for validation
            n_splits: Number of splits for cross-validation
            
        Returns:
            Dictionary with training results for all models
        """
        logger.info("Training ensemble of climate models...")
        
        results = {}
        estimators = []
        
        # Train each base model
        for model_name, model in self.base_models.items():
            logger.info(f"Training {model_name} model...")
            
            try:
                # Train individual model
                model_results = model.train(df, cross_validate=False)
                results[model_name] = model_results
                
                # Add to ensemble estimators
                estimators.append((model_name, model.model))
                
                logger.info(f"{model_name} - Test R²: {model_results['test_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        if not estimators:
            raise ValueError("No models were successfully trained")
        
        # Create ensemble model
        self.ensemble_model = VotingRegressor(estimators=estimators)
        
        # Prepare data for ensemble training
        X, y = self.base_models[self.model_types[0]].prepare_features(df)
        
        # Use the same scaler as the first base model for consistency
        scaler = self.base_models[self.model_types[0]].scaler
        X_scaled = scaler.transform(X)
        
        # Time series cross-validation
        if use_time_series_split:
            ensemble_scores = self._time_series_cross_validate(X_scaled, y, n_splits)
            results['ensemble'] = ensemble_scores
        
        # Train final ensemble on full data
        self.ensemble_model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate ensemble predictions and metrics
        y_pred_ensemble = self.ensemble_model.predict(X_scaled)
        
        ensemble_metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred_ensemble)),
            'mae': mean_absolute_error(y, y_pred_ensemble),
            'r2': r2_score(y, y_pred_ensemble)
        }
        
        results['ensemble_final'] = ensemble_metrics
        
        logger.info(f"Ensemble training completed. Final R²: {ensemble_metrics['r2']:.4f}")
        return results
    
    def _time_series_cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_splits: int
    ) -> Dict[str, float]:
        """Perform time series cross-validation."""
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {'rmse': [], 'mae': [], 'r2': []}
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train ensemble on fold
            ensemble_fold = VotingRegressor(
                estimators=[(name, model.model) for name, model in self.base_models.items()]
            )
            ensemble_fold.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = ensemble_fold.predict(X_val)
            
            # Calculate metrics
            scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            scores['mae'].append(mean_absolute_error(y_val, y_pred))
            scores['r2'].append(r2_score(y_val, y_pred))
        
        # Return mean and std of scores
        return {
            'cv_rmse_mean': np.mean(scores['rmse']),
            'cv_rmse_std': np.std(scores['rmse']),
            'cv_mae_mean': np.mean(scores['mae']),
            'cv_mae_std': np.std(scores['mae']),
            'cv_r2_mean': np.mean(scores['r2']),
            'cv_r2_std': np.std(scores['r2'])
        }
    
    def predict_with_uncertainty(
        self, 
        df: pd.DataFrame,
        return_individual: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            df: DataFrame with features for prediction
            return_individual: Whether to return individual model predictions
            
        Returns:
            Dictionary with ensemble prediction and uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Get predictions from each base model
        individual_predictions = {}
        
        for model_name, model in self.base_models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(df)
                    individual_predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_name}: {e}")
                    continue
        
        if not individual_predictions:
            raise ValueError("No base models available for prediction")
        
        # Calculate ensemble statistics
        predictions_array = np.array(list(individual_predictions.values()))
        
        ensemble_mean = np.mean(predictions_array, axis=0)
        ensemble_std = np.std(predictions_array, axis=0)
        
        # Prediction intervals (assuming normal distribution)
        confidence_95_lower = ensemble_mean - 1.96 * ensemble_std
        confidence_95_upper = ensemble_mean + 1.96 * ensemble_std
        
        results = {
            'prediction': ensemble_mean,
            'uncertainty_std': ensemble_std,
            'confidence_95_lower': confidence_95_lower,
            'confidence_95_upper': confidence_95_upper
        }
        
        if return_individual:
            results['individual_predictions'] = individual_predictions
        
        return results
    
    def predict_future_with_uncertainty(
        self,
        start_year: int,
        end_year: int,
        base_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Predict future temperatures with uncertainty quantification.
        
        Args:
            start_year: Starting year for predictions
            end_year: Ending year for predictions
            base_data: Historical data for context
            
        Returns:
            DataFrame with future predictions and uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Create future date range
        future_dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                future_dates.append({'year': year, 'month': month})
        
        future_df = pd.DataFrame(future_dates)
        
        # Get predictions with uncertainty
        predictions = self.predict_with_uncertainty(future_df, return_individual=True)
        
        # Add predictions to dataframe
        future_df['predicted_temperature'] = predictions['prediction']
        future_df['uncertainty_std'] = predictions['uncertainty_std']
        future_df['confidence_95_lower'] = predictions['confidence_95_lower']
        future_df['confidence_95_upper'] = predictions['confidence_95_upper']
        
        # Add individual model predictions
        for model_name, pred in predictions['individual_predictions'].items():
            future_df[f'{model_name}_prediction'] = pred
        
        return future_df
    
    def evaluate_models(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test data.
        
        Args:
            test_df: Test data DataFrame
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        results = {}
        
        # Get true values
        temp_col = self._get_temperature_column(test_df)
        if not temp_col:
            raise ValueError("No temperature column found in test data")
        
        y_true = test_df[temp_col].values
        
        # Evaluate each base model
        for model_name, model in self.base_models.items():
            if model.is_fitted:
                try:
                    y_pred = model.predict(test_df)
                    
                    results[model_name] = {
                        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'mae': mean_absolute_error(y_true, y_pred),
                        'r2': r2_score(y_true, y_pred)
                    }
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
        
        # Evaluate ensemble
        if self.ensemble_model:
            try:
                ensemble_pred = self.predict_with_uncertainty(test_df)['prediction']
                
                results['ensemble'] = {
                    'rmse': np.sqrt(mean_squared_error(y_true, ensemble_pred)),
                    'mae': mean_absolute_error(y_true, ensemble_pred),
                    'r2': r2_score(y_true, ensemble_pred)
                }
            except Exception as e:
                logger.error(f"Error evaluating ensemble: {e}")
        
        return results
    
    def feature_importance_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature importance across models.
        
        Returns:
            Dictionary with feature importance for each model
        """
        importance_results = {}
        
        for model_name, model in self.base_models.items():
            if model.is_fitted and hasattr(model.model, 'feature_importances_'):
                importance_dict = dict(zip(
                    model.feature_names, 
                    model.model.feature_importances_
                ))
                importance_results[model_name] = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
        
        return importance_results
    
    def save_ensemble(self, save_dir: str) -> None:
        """
        Save the entire ensemble to disk.
        
        Args:
            save_dir: Directory to save the ensemble
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before saving")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each base model
        for model_name, model in self.base_models.items():
            if model.is_fitted:
                model_path = save_path / f"{model_name}_model.joblib"
                model.save_model(str(model_path))
        
        # Save ensemble metadata
        ensemble_data = {
            'model_types': self.model_types,
            'is_fitted': self.is_fitted,
            'ensemble_model': self.ensemble_model
        }
        
        ensemble_path = save_path / "ensemble_metadata.joblib"
        joblib.dump(ensemble_data, ensemble_path)
        
        logger.info(f"Ensemble saved to {save_dir}")
    
    def load_ensemble(self, save_dir: str) -> None:
        """
        Load the entire ensemble from disk.
        
        Args:
            save_dir: Directory containing the saved ensemble
        """
        save_path = Path(save_dir)
        
        # Load ensemble metadata
        ensemble_path = save_path / "ensemble_metadata.joblib"
        ensemble_data = joblib.load(ensemble_path)
        
        self.model_types = ensemble_data['model_types']
        self.is_fitted = ensemble_data['is_fitted']
        self.ensemble_model = ensemble_data['ensemble_model']
        
        # Load each base model
        self.base_models = {}
        for model_type in self.model_types:
            model_path = save_path / f"{model_type}_model.joblib"
            if model_path.exists():
                model = BaselineModel(model_type)
                model.load_model(str(model_path))
                self.base_models[model_type] = model
        
        logger.info(f"Ensemble loaded from {save_dir}")
    
    def _get_temperature_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the temperature column in the dataframe."""
        temp_columns = ['temperature', 'avg_temperature', 'mean_temperature', 'temp']
        
        for col in temp_columns:
            if col in df.columns:
                return col
        
        # Look for columns containing 'temp'
        for col in df.columns:
            if 'temp' in col.lower():
                return col
        
        return None


class IntensityPredictor:
    """
    Predicts the intensity of extreme weather events using ensemble ML models.
    
    Uses features like temperature, humidity, wind speed, and pressure to 
    predict event intensity on a 0-10 scale.
    """
    
    # Feature columns used for intensity prediction
    FEATURE_COLUMNS = [
        'temperature_hourly', 'rain_mm', 'wind_speed_kmh', 'humidity_pct',
        'pressure_hpa', 'month', 'hour', 'latitude_numeric'
    ]
    
    # Climate zone encoding
    CLIMATE_ZONES = ['Tropical', 'Subtropical', 'Temperate', 'Continental', 'Polar', 'Arid']
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        random_state: int = 42
    ) -> None:
        """
        Initialize the intensity predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'ensemble')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.training_metrics: Dict[str, float] = {}
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ML model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=self.random_state
            )
        elif self.model_type == 'ensemble':
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=self.random_state,
                n_jobs=-1
            )
            gb = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state
            )
            self.model = VotingRegressor(
                estimators=[('rf', rf), ('gb', gb)],
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized IntensityPredictor with {self.model_type} model")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare feature matrix from DataFrame.
        
        Args:
            df: Input DataFrame with weather data
        
        Returns:
            Tuple of (feature_matrix, target_values)
        """
        df = df.copy()
        feature_cols = []
        
        # Numeric features
        for col in self.FEATURE_COLUMNS:
            if col in df.columns:
                feature_cols.append(col)
            elif col == 'latitude_numeric' and 'Latitude' in df.columns:
                # Convert latitude to numeric if needed
                df['latitude_numeric'] = pd.to_numeric(
                    df['Latitude'].astype(str).str.replace(r'[NS]', '', regex=True),
                    errors='coerce'
                )
                # Make South negative
                if 'Latitude' in df.columns:
                    south_mask = df['Latitude'].astype(str).str.contains('S', na=False)
                    df.loc[south_mask, 'latitude_numeric'] *= -1
                feature_cols.append('latitude_numeric')
        
        # Climate zone encoding (one-hot)
        if 'climate_zone' in df.columns:
            # Fit label encoder if not fitted
            if not hasattr(self.label_encoder, 'classes_'):
                self.label_encoder.fit(self.CLIMATE_ZONES)
            
            # One-hot encode climate zones
            for zone in self.CLIMATE_ZONES:
                col_name = f'climate_zone_{zone}'
                df[col_name] = (df['climate_zone'] == zone).astype(int)
                feature_cols.append(col_name)
        
        # Event type encoding
        if 'event_type' in df.columns:
            event_types = ['heatwave', 'cold_snap', 'drought', 'extreme_precipitation', 'hurricane', 'tornado']
            for event in event_types:
                col_name = f'event_{event}'
                df[col_name] = (df['event_type'] == event).astype(int)
                feature_cols.append(col_name)
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Extract features
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Get target if available
        y = None
        if 'event_intensity' in df.columns:
            y = df['event_intensity'].values
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        tune_hyperparameters: bool = False,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the intensity prediction model.
        
        Args:
            df: Training DataFrame with event_intensity column
            tune_hyperparameters: Whether to perform grid search
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training intensity prediction model...")
        
        # Filter to only events with intensity > 0
        event_df = df[df['event_intensity'] > 0].copy() if 'event_intensity' in df.columns else df
        
        if len(event_df) < 100:
            logger.warning(f"Limited training data: only {len(event_df)} samples with events")
        
        # Prepare features
        X, y = self.prepare_features(event_df)
        
        if y is None:
            raise ValueError("No event_intensity column found in training data")
        
        logger.info(f"Training on {len(y)} samples with {len(self.feature_names)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Hyperparameter tuning
        if tune_hyperparameters and self.model_type == 'random_forest':
            logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv_folds,
                scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            logger.info(f"Cross-validation RMSE: {cv_rmse:.4f} (+/- {cv_scores.std():.4f})")
        
        # Train final model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        
        self.training_metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'n_samples': len(y),
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"Training completed. R²: {self.training_metrics['r2']:.4f}")
        
        return self.training_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict event intensity for new data.
        
        Args:
            df: DataFrame with weather features
        
        Returns:
            Array of predicted intensity values (0-10 scale)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range
        predictions = np.clip(predictions, 0, 10)
        
        return predictions
    
    def predict_with_features(
        self,
        temperature: float,
        wind_speed: float,
        humidity: float,
        pressure: float,
        rain: float = 0,
        month: int = 6,
        hour: int = 12,
        latitude: float = 0,
        climate_zone: str = 'Temperate',
        event_type: str = 'none'
    ) -> Dict[str, Any]:
        """
        Predict intensity from individual feature values.
        
        Args:
            temperature: Temperature in °C
            wind_speed: Wind speed in km/h
            humidity: Humidity percentage
            pressure: Pressure in hPa
            rain: Precipitation in mm
            month: Month (1-12)
            hour: Hour of day (0-23)
            latitude: Latitude in degrees
            climate_zone: Climate zone name
            event_type: Type of extreme event
        
        Returns:
            Dictionary with prediction and confidence
        """
        # Create single-row DataFrame
        data = {
            'temperature_hourly': temperature,
            'wind_speed_kmh': wind_speed,
            'humidity_pct': humidity,
            'pressure_hpa': pressure,
            'rain_mm': rain,
            'month': month,
            'hour': hour,
            'Latitude': f"{abs(latitude)}{'N' if latitude >= 0 else 'S'}",
            'climate_zone': climate_zone,
            'event_type': event_type
        }
        
        df = pd.DataFrame([data])
        
        # Predict
        intensity = self.predict(df)[0]
        
        # Get feature importance for this prediction
        importance = self.get_feature_importance()
        
        return {
            'predicted_intensity': float(intensity),
            'intensity_category': self._get_intensity_category(intensity),
            'feature_importance': importance
        }
    
    def _get_intensity_category(self, intensity: float) -> str:
        """Get categorical intensity label."""
        if intensity < 2:
            return 'Minor'
        elif intensity < 4:
            return 'Moderate'
        elif intensity < 6:
            return 'Significant'
        elif intensity < 8:
            return 'Severe'
        else:
            return 'Extreme'
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # For ensemble models, average importance
            importance = np.zeros(len(self.feature_names))
            for name, estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importance += estimator.feature_importances_
            importance /= len(self.model.estimators_)
        else:
            return {}
        
        return dict(sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_type': self.model_type
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        save_data = joblib.load(filepath)
        
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.label_encoder = save_data['label_encoder']
        self.feature_names = save_data['feature_names']
        self.training_metrics = save_data['training_metrics']
        self.model_type = save_data['model_type']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


def train_intensity_model(
    input_path: str,
    output_path: str,
    model_type: str = 'random_forest',
    tune_hyperparameters: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to train an intensity prediction model.
    
    Args:
        input_path: Path to synthetic data parquet file
        output_path: Path to save trained model
        model_type: Type of ML model to use
        tune_hyperparameters: Whether to tune hyperparameters
    
    Returns:
        Training metrics dictionary
    """
    logger.info(f"Training intensity model from {input_path}")
    
    # Load data
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Initialize and train model
    predictor = IntensityPredictor(model_type=model_type)
    metrics = predictor.train(df, tune_hyperparameters=tune_hyperparameters)
    
    # Save model
    predictor.save_model(output_path)
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    
    return {
        'metrics': metrics,
        'feature_importance': importance,
        'model_path': output_path
    }