"""
Climate predictor with ensemble methods and advanced features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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