# src/amgeo/ml/classifier.py
"""
Advanced ensemble classifier for aquifer detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Comprehensive prediction result with uncertainty"""

    aquifer_probability: float
    binary_prediction: int
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float
    confidence_interval: Tuple[float, float]
    individual_predictions: Dict[str, float]
    prediction_quality: str
    recommendations: List[str]


class AquiferClassificationEnsemble:
    """
    Professional ensemble classifier with uncertainty quantification
    """

    def __init__(self, settings=None, random_state=42):
        self.settings = settings or get_settings()
        self.ml_config = self.settings.ml
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.models = {}
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance_ = None

        self._initialize_models()

    def _initialize_models(self):
        """Initialize ensemble models with optimized parameters"""
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=self.ml_config.n_estimators,
                max_depth=self.ml_config.max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=self.random_state,
            ),
            "svm_rbf": SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=self.random_state,
                gamma="scale",
            ),
            "neural_network": MLPClassifier(
                hidden_layer_sizes=(50, 25),
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        }

        # Wrap models with calibration for better probability estimates
        for name, model in self.models.items():
            self.models[name] = CalibratedClassifierCV(model, method="isotonic", cv=3)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Train the ensemble with cross-validation

        Args:
            X: Feature matrix
            y: Target labels (0=no aquifer, 1=aquifer)
            feature_names: Optional feature names
        """
        logger.info(f"Training ensemble on {len(X)} samples with {X.shape[1]} features")

        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train each model with cross-validation
        cv_scores = {}

        for name, model in self.models.items():
            try:
                # Cross-validation
                cv = StratifiedKFold(
                    n_splits=self.ml_config.default_cv_folds,
                    shuffle=True,
                    random_state=self.random_state,
                )
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
                cv_scores[name] = scores

                # Fit on full dataset
                model.fit(X_scaled, y)

                logger.info(
                    f"{name}: CV AUC = {scores.mean():.3f} ± {scores.std():.3f}"
                )

            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                del self.models[name]

        # Calculate ensemble feature importance (from Random Forest)
        if "random_forest" in self.models:
            try:
                # Access the base estimator's feature importance
                rf_model = (
                    self.models["random_forest"]
                    .calibrated_classifiers_[0]
                    .base_estimator
                )
                self.feature_importance_ = rf_model.feature_importances_
            except:
                logger.warning("Could not extract feature importance")
                self.feature_importance_ = np.ones(X.shape[1]) / X.shape[1]

        self.is_fitted = True
        logger.info(f"Ensemble training completed with {len(self.models)} models")

        return cv_scores

    def predict_with_uncertainty(self, X: np.ndarray) -> PredictionResult:
        """
        Predict with comprehensive uncertainty quantification

        Args:
            X: Feature matrix (single sample or multiple samples)

        Returns:
            PredictionResult with detailed uncertainty analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        individual_predictions = {}
        probabilities = []

        for name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(X_scaled)
                if pred_proba.shape[1] > 1:
                    prob = pred_proba[:, 1]  # Probability of positive class
                else:
                    prob = pred_proba[:, 0]

                individual_predictions[name] = float(prob[0])
                probabilities.append(prob[0])

            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        if not probabilities:
            raise RuntimeError("No models available for prediction")

        probabilities = np.array(probabilities)

        # Ensemble statistics
        ensemble_prob = np.mean(probabilities)
        ensemble_std = np.std(probabilities)

        # Epistemic uncertainty (model disagreement)
        epistemic_uncertainty = ensemble_std

        # Aleatoric uncertainty (estimate based on prediction confidence)
        # Higher uncertainty for predictions near 0.5
        aleatoric_uncertainty = 0.1 * (1 - 2 * abs(ensemble_prob - 0.5))

        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

        # Confidence interval (95%)
        z_score = 1.96
        ci_lower = max(0, ensemble_prob - z_score * total_uncertainty)
        ci_upper = min(1, ensemble_prob + z_score * total_uncertainty)

        # Binary prediction
        binary_prediction = int(ensemble_prob >= 0.5)

        # Prediction quality assessment
        if total_uncertainty < 0.1:
            quality = "High"
        elif total_uncertainty < 0.2:
            quality = "Medium"
        else:
            quality = "Low"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            ensemble_prob, total_uncertainty, epistemic_uncertainty
        )

        return PredictionResult(
            aquifer_probability=float(ensemble_prob),
            binary_prediction=binary_prediction,
            epistemic_uncertainty=float(epistemic_uncertainty),
            aleatoric_uncertainty=float(aleatoric_uncertainty),
            total_uncertainty=float(total_uncertainty),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            individual_predictions=individual_predictions,
            prediction_quality=quality,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, probability: float, total_uncertainty: float, epistemic_uncertainty: float
    ) -> List[str]:
        """Generate professional recommendations based on prediction"""
        recommendations = []

        # Main drilling recommendation
        if probability >= 0.8 and total_uncertainty < 0.15:
            recommendations.append(
                "Strong drilling recommendation - high probability with low uncertainty"
            )
        elif probability >= 0.6:
            recommendations.append(
                "Drilling recommended - good aquifer potential identified"
            )
        elif probability >= 0.4:
            recommendations.append(
                "Further investigation recommended before drilling decision"
            )
        else:
            recommendations.append("Drilling not recommended - low aquifer potential")

        # Uncertainty-based recommendations
        if total_uncertainty > 0.25:
            recommendations.append(
                "High prediction uncertainty - consider additional VES measurements"
            )

        if epistemic_uncertainty > 0.2:
            recommendations.append(
                "High model uncertainty - additional training data recommended"
            )

        # Risk assessment
        if probability > 0.5 and total_uncertainty > 0.3:
            recommendations.append(
                "Moderate potential but high uncertainty - assess drilling risks carefully"
            )

        return recommendations

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance scores"""
        if self.feature_importance_ is None or self.feature_names is None:
            return None

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.feature_importance_}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save_model(self, filepath: str):
        """Save the trained ensemble model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            "models": self.models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_importance_": self.feature_importance_,
            "is_fitted": self.is_fitted,
            "random_state": self.random_state,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained ensemble model"""
        model_data = joblib.load(filepath)

        self.models = model_data["models"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.feature_importance_ = model_data.get("feature_importance_")
        self.is_fitted = model_data["is_fitted"]
        self.random_state = model_data.get("random_state", 42)

        logger.info(f"Model loaded from {filepath}")
