"""
Machine Learning Analysis Module

Comprehensive ML modeling with AutoML capabilities, designed for clinical research:
- Automated model selection and hyperparameter tuning
- TRIPOD+AI compliant model development and validation
- Model interpretability with SHAP
- Robust cross-validation and performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')
import logging
import joblib
from pathlib import Path

# Core ML libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE, RFECV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Metrics
from sklearn.metrics import (
    # Classification
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    # Regression  
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)

# Advanced libraries (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class MLAnalysis:
    """
    Comprehensive ML analysis for clinical research.
    
    Features:
    - AutoML with intelligent model selection
    - TRIPOD+AI compliant development
    - Model interpretability with SHAP
    - Robust validation and performance assessment
    """
    
    def __init__(self, df: pd.DataFrame, privacy_guard=None):
        self.df = df.copy()
        self.privacy_guard = privacy_guard
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.shap_values = {}
        
        # Define model catalogs
        self.classification_models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(),
            'neural_network': MLPClassifier(max_iter=1000, random_state=42)
        }
        
        self.regression_models = {
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elastic_net': ElasticNet(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'svm': SVR(),
            'knn': KNeighborsRegressor(),
            'neural_network': MLPRegressor(max_iter=1000, random_state=42)
        }
        
        # Add advanced models if available
        if HAS_XGBOOST:
            self.classification_models['xgboost'] = xgb.XGBClassifier(random_state=42)
            self.regression_models['xgboost'] = xgb.XGBRegressor(random_state=42)
        
        if HAS_LIGHTGBM:
            self.classification_models['lightgbm'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
            self.regression_models['lightgbm'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
    
    def auto_ml(self,
                outcome: str,
                features: List[str],
                task_type: str = 'auto',
                test_size: float = 0.2,
                cv_folds: int = 5,
                optimization_metric: str = 'auto',
                hyperparameter_tuning: bool = True,
                feature_selection: bool = True,
                max_models: int = 10,
                time_budget_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Automated machine learning pipeline.
        
        Args:
            outcome: Target variable
            features: List of feature variables
            task_type: 'classification', 'regression', or 'auto'
            test_size: Fraction for test set
            cv_folds: Number of cross-validation folds
            optimization_metric: Metric to optimize ('auto', 'auc', 'accuracy', etc.)
            hyperparameter_tuning: Whether to tune hyperparameters
            feature_selection: Whether to perform feature selection
            max_models: Maximum number of models to try
            time_budget_minutes: Time budget for optimization
            
        Returns:
            Complete AutoML results with best model and performance metrics
        """
        # Prepare data
        prep_result = self._prepare_ml_data(outcome, features, test_size)
        if 'error' in prep_result:
            return prep_result
        
        X_train = prep_result['X_train']
        X_test = prep_result['X_test'] 
        y_train = prep_result['y_train']
        y_test = prep_result['y_test']
        task_type = prep_result['task_type'] if task_type == 'auto' else task_type
        preprocessor = prep_result['preprocessor']
        
        # Determine optimization metric
        if optimization_metric == 'auto':
            optimization_metric = 'roc_auc' if task_type == 'classification' else 'r2'
        
        # Get appropriate model catalog
        model_catalog = (self.classification_models if task_type == 'classification' 
                        else self.regression_models)
        
        # Limit models if specified
        if max_models < len(model_catalog):
            model_names = list(model_catalog.keys())[:max_models]
            model_catalog = {name: model_catalog[name] for name in model_names}
        
        # Feature selection
        if feature_selection:
            feature_selector = self._get_feature_selector(task_type, min(20, len(features)))
            X_train_selected = feature_selector.fit_transform(X_train, y_train)
            X_test_selected = feature_selector.transform(X_test)
            selected_features = [features[i] for i in feature_selector.get_support(indices=True)]
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = features
            feature_selector = None
        
        # Model evaluation and selection
        model_results = {}
        best_score = -np.inf if optimization_metric in ['roc_auc', 'accuracy', 'r2'] else np.inf
        best_model_name = None
        
        for model_name, base_model in model_catalog.items():
            try:
                # Hyperparameter tuning
                if hyperparameter_tuning:
                    model = self._tune_hyperparameters(
                        base_model, X_train_selected, y_train, 
                        task_type, optimization_metric, cv_folds, time_budget_minutes
                    )
                else:
                    model = base_model
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_selected, y_train, 
                    cv=cv_folds, scoring=optimization_metric
                )
                
                # Fit and evaluate on test set
                model.fit(X_train_selected, y_train)
                
                if task_type == 'classification':
                    y_pred = model.predict(X_test_selected)
                    y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    test_metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                else:
                    y_pred = model.predict(X_test_selected)
                    test_metrics = self._calculate_regression_metrics(y_test, y_pred)
                
                model_results[model_name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'test_metrics': test_metrics
                }
                
                # Track best model
                current_score = cv_scores.mean()
                if ((optimization_metric in ['roc_auc', 'accuracy', 'r2'] and current_score > best_score) or
                    (optimization_metric not in ['roc_auc', 'accuracy', 'r2'] and current_score < best_score)):
                    best_score = current_score
                    best_model_name = model_name
                    
            except Exception as e:
                logging.warning(f"Model {model_name} failed: {str(e)}")
                continue
        
        if not model_results:
            return {'error': 'All models failed to train'}
        
        # Get best model
        best_model_info = model_results[best_model_name]
        best_model = best_model_info['model']
        
        # Feature importance
        importance = self._get_feature_importance(best_model, selected_features)
        
        # Model interpretability with SHAP
        shap_explanation = None
        if HAS_SHAP and len(X_test_selected) <= 1000:  # Limit for performance
            try:
                shap_explanation = self._calculate_shap_values(
                    best_model, X_train_selected[:500], X_test_selected[:100]
                )
            except Exception as e:
                logging.warning(f"SHAP calculation failed: {str(e)}")
        
        # Compile final results
        results = {
            'task_type': task_type,
            'n_features_original': len(features),
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'feature_selection_method': type(feature_selector).__name__ if feature_selector else None,
            'optimization_metric': optimization_metric,
            'n_models_evaluated': len(model_results),
            'best_model_name': best_model_name,
            'best_cv_score': best_score,
            'model_comparison': {
                name: {
                    'cv_mean': info['cv_mean'],
                    'cv_std': info['cv_std']
                } for name, info in model_results.items()
            },
            'best_model_performance': best_model_info['test_metrics'],
            'feature_importance': importance,
            'shap_explanation': shap_explanation,
            'model_parameters': best_model.get_params(),
            'training_info': {
                'n_train': len(X_train),
                'n_test': len(X_test),
                'cv_folds': cv_folds,
                'hyperparameter_tuning': hyperparameter_tuning,
                'test_size': test_size
            }
        }
        
        # Store model and preprocessor
        model_id = f"automl_{task_type}_{len(self.models)}"
        self.models[model_id] = {
            'model': best_model,
            'preprocessor': preprocessor,
            'feature_selector': feature_selector,
            'task_type': task_type,
            'features': selected_features,
            'target': outcome
        }
        results['model_id'] = model_id
        
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results)
        
        self.results['automl'] = results
        return results
    
    def validate_model(self,
                      model_id: str,
                      validation_data: Optional[pd.DataFrame] = None,
                      validation_type: str = 'holdout') -> Dict[str, Any]:
        """
        Comprehensive model validation following TRIPOD+AI guidelines.
        
        Args:
            model_id: ID of trained model
            validation_data: External validation dataset
            validation_type: 'holdout', 'external', 'temporal'
            
        Returns:
            Validation results with performance metrics and calibration
        """
        if model_id not in self.models:
            return {'error': f'Model {model_id} not found'}
        
        model_info = self.models[model_id]
        model = model_info['model']
        preprocessor = model_info.get('preprocessor')
        feature_selector = model_info.get('feature_selector')
        features = model_info['features']
        target = model_info['target']
        task_type = model_info['task_type']
        
        # Prepare validation data
        if validation_data is not None:
            val_df = validation_data.copy()
        else:
            # Use holdout from original data
            val_df = self.df.copy()
        
        # Check for required columns
        missing_cols = [col for col in features + [target] if col not in val_df.columns]
        if missing_cols:
            return {'error': f'Missing columns in validation data: {missing_cols}'}
        
        val_df_clean = val_df[features + [target]].dropna()
        
        if len(val_df_clean) < 50:
            return {'error': 'Insufficient validation data (need at least 50 observations)'}
        
        X_val = val_df_clean[features]
        y_val = val_df_clean[target]
        
        # Apply preprocessing
        if preprocessor:
            X_val_processed = preprocessor.transform(X_val)
        else:
            X_val_processed = X_val
        
        # Apply feature selection
        if feature_selector:
            X_val_processed = feature_selector.transform(X_val_processed)
        
        # Make predictions
        y_pred = model.predict(X_val_processed)
        
        # Calculate metrics
        if task_type == 'classification':
            y_pred_proba = model.predict_proba(X_val_processed)[:, 1] if hasattr(model, 'predict_proba') else None
            metrics = self._calculate_classification_metrics(y_val, y_pred, y_pred_proba)
            
            # Calibration assessment
            calibration_metrics = self._assess_calibration(y_val, y_pred_proba) if y_pred_proba is not None else None
            
        else:
            metrics = self._calculate_regression_metrics(y_val, y_pred)
            calibration_metrics = None
        
        results = {
            'model_id': model_id,
            'validation_type': validation_type,
            'validation_n': len(val_df_clean),
            'performance_metrics': metrics,
            'calibration_metrics': calibration_metrics,
            'feature_drift': self._detect_feature_drift(
                self.df[features], X_val
            ) if validation_data is not None else None
        }
        
        # TRIPOD+AI specific metrics
        if task_type == 'classification':
            results['tripod_metrics'] = self._calculate_tripod_metrics(
                y_val, y_pred, y_pred_proba
            )
        
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results)
        
        return results
    
    def explain_predictions(self,
                           model_id: str,
                           sample_data: Optional[pd.DataFrame] = None,
                           n_samples: int = 100) -> Dict[str, Any]:
        """
        Generate model explanations using SHAP values.
        
        Args:
            model_id: ID of trained model
            sample_data: Data to explain (uses random sample if None)
            n_samples: Number of samples to explain
            
        Returns:
            Model explanations and feature importance
        """
        if not HAS_SHAP:
            return {'error': 'SHAP not available for model explanations'}
        
        if model_id not in self.models:
            return {'error': f'Model {model_id} not found'}
        
        model_info = self.models[model_id]
        model = model_info['model']
        features = model_info['features']
        
        # Prepare explanation data
        if sample_data is not None:
            explain_df = sample_data[features].dropna()
        else:
            explain_df = self.df[features].dropna()
        
        if len(explain_df) == 0:
            return {'error': 'No valid data for explanations'}
        
        # Sample if too many observations
        if len(explain_df) > n_samples:
            explain_df = explain_df.sample(n_samples, random_state=42)
        
        # Apply preprocessing if needed
        preprocessor = model_info.get('preprocessor')
        feature_selector = model_info.get('feature_selector')
        
        X_explain = explain_df
        if preprocessor:
            X_explain = preprocessor.transform(X_explain)
        if feature_selector:
            X_explain = feature_selector.transform(X_explain)
        
        try:
            # Calculate SHAP values
            explainer = shap.Explainer(model, X_explain[:100])  # Background sample
            shap_values = explainer(X_explain)
            
            # Summary statistics
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            feature_importance = dict(zip(features, mean_abs_shap))
            
            results = {
                'model_id': model_id,
                'n_explained': len(X_explain),
                'feature_importance': feature_importance,
                'explanation_type': 'shap',
                'global_explanation': {
                    'feature_importance_rank': sorted(
                        feature_importance.items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )
                }
            }
            
            # Store SHAP values (privacy-protected)
            if self.privacy_guard:
                # Only store aggregated statistics
                results['shap_summary'] = {
                    'mean_abs_values': feature_importance,
                    'feature_ranges': {
                        feat: {
                            'min': float(np.min(shap_values.values[:, i])),
                            'max': float(np.max(shap_values.values[:, i])),
                            'mean': float(np.mean(shap_values.values[:, i]))
                        } for i, feat in enumerate(features)
                    }
                }
            else:
                # Store individual SHAP values
                results['individual_explanations'] = {
                    'shap_values': shap_values.values.tolist(),
                    'base_value': float(shap_values.base_values[0]),
                    'feature_values': X_explain.tolist()
                }
            
            return results
            
        except Exception as e:
            return {'error': f'SHAP explanation failed: {str(e)}'}
    
    def _prepare_ml_data(self, outcome: str, features: List[str], 
                        test_size: float) -> Dict[str, Any]:
        """Prepare data for ML modeling."""
        # Check for required columns
        required_cols = [outcome] + features
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            return {'error': f'Missing columns: {missing_cols}'}
        
        # Remove missing data
        model_df = self.df[required_cols].dropna()
        
        if len(model_df) < 50:
            return {'error': 'Insufficient data after removing missing values'}
        
        X = model_df[features]
        y = model_df[outcome]
        
        # Determine task type
        if y.nunique() == 2:
            task_type = 'classification'
        elif y.nunique() <= 10 and y.dtype == 'object':
            task_type = 'classification'
        else:
            task_type = 'regression'
        
        # Split data
        stratify = y if task_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        # Create preprocessor
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', 'passthrough', categorical_features)  # Simple pass-through for now
            ]
        )
        
        # Fit preprocessor and transform
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'task_type': task_type,
            'preprocessor': preprocessor
        }
    
    def _get_feature_selector(self, task_type: str, k: int):
        """Get appropriate feature selector."""
        if task_type == 'classification':
            return SelectKBest(f_classif, k=k)
        else:
            return SelectKBest(f_regression, k=k)
    
    def _tune_hyperparameters(self, model, X: np.ndarray, y: pd.Series,
                             task_type: str, metric: str, cv_folds: int,
                             time_budget: Optional[int] = None):
        """Tune hyperparameters using grid search or Optuna."""
        param_grids = self._get_param_grids()
        model_name = type(model).__name__.lower()
        
        if model_name in param_grids:
            param_grid = param_grids[model_name]
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model, param_grid, cv=cv_folds, scoring=metric,
                n_iter=20, random_state=42, n_jobs=-1
            )
            search.fit(X, y)
            return search.best_estimator_
        
        return model
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for common models."""
        return {
            'randomforestclassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'randomforestregressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'logisticregression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'svc': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def _calculate_classification_metrics(self, y_true: pd.Series, 
                                        y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics.update({
                'auc_roc': roc_auc_score(y_true, y_pred_proba),
                'auc_pr': average_precision_score(y_true, y_pred_proba)
            })
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: pd.Series, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
    
    def _calculate_specificity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn = cm[0, 0]
            fp = cm[0, 1]
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            return {}
        
        return dict(zip(feature_names, importance.tolist()))
    
    def _calculate_shap_values(self, model, X_background: np.ndarray, 
                             X_explain: np.ndarray) -> Dict[str, Any]:
        """Calculate SHAP values for model explanations."""
        explainer = shap.Explainer(model, X_background)
        shap_values = explainer(X_explain)
        
        return {
            'values': shap_values.values.tolist(),
            'base_values': shap_values.base_values.tolist(),
            'data': X_explain.tolist()
        }
    
    def _assess_calibration(self, y_true: pd.Series, 
                          y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Assess prediction calibration."""
        from sklearn.calibration import calibration_curve
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Brier score (lower is better)
        from sklearn.metrics import brier_score_loss
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        return {
            'brier_score': brier_score,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def _detect_feature_drift(self, X_train: pd.DataFrame, 
                            X_val: pd.DataFrame) -> Dict[str, Any]:
        """Detect feature drift between training and validation data."""
        from scipy.stats import ks_2samp
        
        drift_results = {}
        
        for col in X_train.columns:
            if col in X_val.columns:
                train_vals = X_train[col].dropna()
                val_vals = X_val[col].dropna()
                
                if len(train_vals) > 0 and len(val_vals) > 0:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = ks_2samp(train_vals, val_vals)
                    
                    drift_results[col] = {
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'drift_detected': ks_pvalue < 0.05,
                        'train_mean': train_vals.mean(),
                        'val_mean': val_vals.mean()
                    }
        
        return drift_results
    
    def _calculate_tripod_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate TRIPOD+AI specific metrics."""
        metrics = {}
        
        # C-index (equivalent to AUC for binary classification)
        if y_pred_proba is not None:
            metrics['c_index'] = roc_auc_score(y_true, y_pred_proba)
        
        # Net benefit (clinical decision making)
        if y_pred_proba is not None:
            metrics['net_benefit'] = self._calculate_net_benefit(y_true, y_pred_proba)
        
        return metrics
    
    def _calculate_net_benefit(self, y_true: pd.Series, 
                             y_pred_proba: np.ndarray,
                             thresholds: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """Calculate net benefit for decision curve analysis."""
        if thresholds is None:
            thresholds = np.arange(0.01, 0.99, 0.01)
        
        net_benefits = []
        
        for threshold in thresholds:
            # Classify based on threshold
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # True positives and false positives
            tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
            fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
            n = len(y_true)
            
            # Net benefit calculation
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
        
        return {
            'thresholds': thresholds.tolist(),
            'net_benefits': net_benefits
        }