"""
Baseline machine learning models for climate prediction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib


logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Baseline machine learning models for climate prediction.
    
    Provides various regression models for temperature prediction:
    - Linear Regression
    - Ridge Regression  
    - Random Forest
    - Gradient Boosting
    """
    
    def __init__(self, model_type: str = "random_forest") -> None:
        """
        Initialize baseline model.
        
        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the specified model type."""
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = models[self.model_type]
        logger.info(f"Initialized {self.model_type} model")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from climate data.
        
        Args:
            df: DataFrame with climate data
            
        Returns:
            Tuple of (features, target) arrays
        """
        required_cols = ['year', 'month']
        temp_col = self._get_temperature_column(df)
        
        if not temp_col:
            raise ValueError("No temperature column found in the dataset")
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        # Create features
        features_df = pd.DataFrame()
        
        # Time features
        features_df['year'] = df['year']
        features_df['month'] = df['month']
        features_df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        
        # Cyclical encoding for month
        features_df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Seasonal features
        features_df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)
        features_df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
        features_df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        features_df['is_fall'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
        
        # Lag features (if enough data)
        if len(df) > 12:
            features_df['temp_lag_1'] = df[temp_col].shift(1)
            features_df['temp_lag_12'] = df[temp_col].shift(12)  # Same month, previous year
        
        # Rolling averages (if enough data)
        if len(df) > 24:
            features_df['temp_ma_3'] = df[temp_col].rolling(window=3, min_periods=1).mean()
            features_df['temp_ma_12'] = df[temp_col].rolling(window=12, min_periods=1).mean()
        
        # Remove rows with NaN values
        features_df = features_df.fillna(method='bfill').fillna(method='ffill')
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        # Target variable
        target = df[temp_col].values
        
        return features_df.values, target
    
    def train(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        cross_validate: bool = True
    ) -> Dict[str, float]:
        """
        Train the baseline model.
        
        Args:
            df: Training data DataFrame
            test_size: Fraction of data to use for testing
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        # Cross-validation
        if cross_validate:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
            metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            metrics['feature_importance'] = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        
        logger.info(f"Model training completed. Test R²: {metrics['test_r2']:.4f}")
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_future(
        self, 
        start_year: int, 
        end_year: int,
        base_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Predict future temperatures.
        
        Args:
            start_year: Starting year for predictions
            end_year: Ending year for predictions
            base_data: Historical data for lag features
            
        Returns:
            DataFrame with future predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Create future date range
        future_dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                future_dates.append({'year': year, 'month': month})
        
        future_df = pd.DataFrame(future_dates)
        
        # If base_data provided, append it for lag features
        if base_data is not None:
            temp_col = self._get_temperature_column(base_data)
            if temp_col:
                # Create a combined dataframe for feature preparation
                combined_df = pd.concat([
                    base_data[['year', 'month', temp_col]].rename(columns={temp_col: 'temperature'}),
                    future_df.assign(temperature=np.nan)
                ], ignore_index=True)
                
                # For prediction, we'll use the future portion
                future_predictions = []
                
                for i, row in future_df.iterrows():
                    # Create a mini dataset up to this point
                    current_idx = len(base_data) + i
                    current_data = combined_df.iloc[:current_idx + 1].copy()
                    
                    # Fill missing temperature with previous prediction if available
                    if i > 0:
                        current_data.iloc[-1, current_data.columns.get_loc('temperature')] = future_predictions[-1]
                    
                    # Prepare features for this row
                    try:
                        X, _ = self.prepare_features(current_data.iloc[[-1]])
                        X_scaled = self.scaler.transform(X)
                        pred = self.model.predict(X_scaled)[0]
                        future_predictions.append(pred)
                    except:
                        # Fallback: use simple features without lag
                        simple_data = future_df.iloc[[i]].assign(temperature=0)
                        X, _ = self.prepare_features(simple_data)
                        X_scaled = self.scaler.transform(X)
                        pred = self.model.predict(X_scaled)[0]
                        future_predictions.append(pred)
                
                future_df['predicted_temperature'] = future_predictions
            else:
                # Simple prediction without lag features
                future_df['predicted_temperature'] = self.predict(future_df.assign(temperature=0))
        else:
            # Simple prediction without historical data
            future_df['predicted_temperature'] = self.predict(future_df.assign(temperature=0))
        
        return future_df
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {model_path}")
    
    def hyperparameter_tuning(
        self, 
        df: pd.DataFrame,
        param_grid: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            df: Training data DataFrame
            param_grid: Parameter grid for GridSearchCV
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            # Default parameter grids
            param_grids = {
                'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'lasso': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
            param_grid = param_grids.get(self.model_type, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid available for {self.model_type}")
            return {}
        
        logger.info(f"Performing hyperparameter tuning for {self.model_type}...")
        
        # Prepare data
        X, y = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, 
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")
        
        return results
    
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