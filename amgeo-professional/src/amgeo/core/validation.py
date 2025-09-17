# src/amgeo/core/validation.py
"""
Professional VES data validation following industry standards
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats

from amgeo.config.settings import get_settings

logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """Data quality classification levels"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class ValidationResult:
    """Comprehensive validation result"""

    is_valid: bool
    quality_level: DataQualityLevel
    quality_score: float  # 0-100
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class VESDataValidator:
    """
    Professional VES data validator following ASTM D6431 standards
    """

    def __init__(self, settings=None):
        self.settings = get_settings()
        self.inversion_config = self.settings.inversion

        # Industry standard thresholds
        self.thresholds = {
            "min_points": self.inversion_config.min_data_points,
            "max_points": 200,
            "min_ab2": self.inversion_config.min_ab2_spacing,
            "max_ab2": self.inversion_config.max_ab2_spacing,
            "min_rhoa": 0.1,
            "max_rhoa": 100000.0,
            "min_ab2_mn2_ratio": 2.0,
            "max_ab2_mn2_ratio": 100.0,
            "max_noise_level": 0.15,
            "min_data_span": 1.0,  # log decades
            "ideal_data_span": 2.5,
        }

    def validate_ves_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive VES data validation

        Args:
            df: DataFrame with AB2, MN2, Rhoa columns

        Returns:
            ValidationResult with detailed assessment
        """
        issues = []
        warnings = []
        recommendations = []

        # Check required columns
        required_cols = ["AB2", "MN2", "Rhoa"]
        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
            issues.append(f"CRITICAL: Missing required columns: {missing_cols}")
            return ValidationResult(
                is_valid=False,
                quality_level=DataQualityLevel.UNACCEPTABLE,
                quality_score=0,
                issues=issues,
                warnings=warnings,
                recommendations=["Upload file with AB2, MN2, Rhoa columns"],
                metadata={},
            )

        # Extract data arrays
        ab2 = df["AB2"].values
        mn2 = df["MN2"].values
        rhoa = df["Rhoa"].values

        # Basic data integrity checks
        basic_issues = self._check_data_integrity(ab2, mn2, rhoa)
        issues.extend(basic_issues)

        # Configuration validation
        config_issues, config_warnings = self._validate_electrode_configuration(
            ab2, mn2
        )
        issues.extend(config_issues)
        warnings.extend(config_warnings)

        # Data quality assessment
        quality_metrics = self._assess_data_quality(ab2, mn2, rhoa)

        # Noise analysis
        noise_metrics = self._analyze_noise_level(ab2, rhoa)
        if noise_metrics["noise_level"] > self.thresholds["max_noise_level"]:
            warnings.append(
                f"High noise level detected: {noise_metrics['noise_level']:.3f}"
            )

        # Coverage and resolution analysis
        coverage_metrics = self._analyze_coverage_resolution(ab2)

        # Outlier detection
        outlier_indices = self._detect_outliers(rhoa)
        if len(outlier_indices) > 0:
            warnings.append(f"Potential outliers at indices: {outlier_indices}")
            recommendations.append("Review outlier points for measurement errors")

        # Generate recommendations
        recs = self._generate_recommendations(
            ab2,
            rhoa,
            quality_metrics,
            noise_metrics,
            coverage_metrics,
            len(issues),
            len(warnings),
        )
        recommendations.extend(recs)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            len(issues), len(warnings), quality_metrics, noise_metrics, coverage_metrics
        )

        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)

        # Compile metadata
        metadata = {
            "n_points": len(ab2),
            "ab2_range": (float(ab2.min()), float(ab2.max())),
            "rhoa_range": (float(rhoa.min()), float(rhoa.max())),
            "data_span_decades": coverage_metrics["data_span"],
            "measurement_density": coverage_metrics["measurement_density"],
            "noise_level": noise_metrics["noise_level"],
            "smoothness_score": noise_metrics["smoothness_score"],
            **quality_metrics,
        }

        return ValidationResult(
            is_valid=len([i for i in issues if "CRITICAL" in i]) == 0,
            quality_level=quality_level,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metadata=metadata,
        )

    def _check_data_integrity(self, ab2, mn2, rhoa) -> List[str]:
        """Basic data integrity checks"""
        issues = []

        # Check data length
        if len(ab2) < self.thresholds["min_points"]:
            issues.append(
                f"CRITICAL: Insufficient data points ({len(ab2)} < {self.thresholds['min_points']})"
            )

        # Check for non-positive values
        if not np.all(ab2 > 0):
            issues.append("CRITICAL: Non-positive AB2 values detected")
        if not np.all(mn2 > 0):
            issues.append("CRITICAL: Non-positive MN2 values detected")
        if not np.all(rhoa > 0):
            issues.append("CRITICAL: Non-positive resistivity values detected")

        # Check for NaN/infinite values
        if not np.all(np.isfinite(ab2)):
            issues.append("CRITICAL: Non-finite AB2 values detected")
        if not np.all(np.isfinite(mn2)):
            issues.append("CRITICAL: Non-finite MN2 values detected")
        if not np.all(np.isfinite(rhoa)):
            issues.append("CRITICAL: Non-finite resistivity values detected")

        # Check AB2 monotonicity
        if not np.all(np.diff(ab2) >= 0):
            issues.append("AB2 values not monotonically increasing - sort required")

        # Check value ranges
        if np.any(ab2 < self.thresholds["min_ab2"]):
            issues.append(f"AB2 values below minimum ({self.thresholds['min_ab2']} m)")
        if np.any(ab2 > self.thresholds["max_ab2"]):
            issues.append(f"AB2 values above maximum ({self.thresholds['max_ab2']} m)")

        return issues

    def _validate_electrode_configuration(
        self, ab2, mn2
    ) -> Tuple[List[str], List[str]]:
        """Validate electrode configuration per ASTM guidelines"""
        issues = []
        warnings = []

        # AB2/MN2 ratio checks
        ratios = ab2 / mn2

        if np.any(ratios < self.thresholds["min_ab2_mn2_ratio"]):
            issues.append(
                f"AB2/MN2 ratio too small (< {self.thresholds['min_ab2_mn2_ratio']})"
            )

        if np.any(ratios > self.thresholds["max_ab2_mn2_ratio"]):
            warnings.append(
                f"AB2/MN2 ratio very large (> {self.thresholds['max_ab2_mn2_ratio']})"
            )

        # Check spacing ratios between consecutive measurements
        if len(ab2) > 1:
            spacing_ratios = ab2[1:] / ab2[:-1]

            if np.any(spacing_ratios < 1.2):
                warnings.append("AB2 spacing ratios too small - limited resolution")
            if np.any(spacing_ratios > 5.0):
                warnings.append("AB2 spacing ratios too large - may miss features")

        return issues, warnings

    def _assess_data_quality(self, ab2, mn2, rhoa) -> Dict[str, float]:
        """Assess various data quality metrics"""
        metrics = {}

        # Data span in log decades
        metrics["data_span"] = np.log10(ab2.max() / ab2.min())

        # Resistivity range
        metrics["resistivity_range"] = np.log10(rhoa.max() / rhoa.min())

        # Measurement density (points per log decade)
        metrics["measurement_density"] = (
            len(ab2) / metrics["data_span"] if metrics["data_span"] > 0 else 0
        )

        # Coefficient of variation
        metrics["cv_resistivity"] = np.std(rhoa) / np.mean(rhoa)

        # AB2/MN2 ratio statistics
        ratios = ab2 / mn2
        metrics["ab2_mn2_ratio_mean"] = np.mean(ratios)
        metrics["ab2_mn2_ratio_std"] = np.std(ratios)

        return metrics

    def _analyze_noise_level(self, ab2, rhoa) -> Dict[str, float]:
        """Analyze noise level in VES data"""
        metrics = {}

        if len(rhoa) < 3:
            return {"noise_level": 0.0, "smoothness_score": 1.0}

        # Calculate local variability as noise proxy
        log_rhoa = np.log10(rhoa)

        # Second derivative as noise indicator
        second_derivative = np.diff(log_rhoa, n=2)
        metrics["noise_level"] = np.std(second_derivative)

        # Smoothness via correlation with smoothed version
        if len(rhoa) >= 5:
            window = min(5, len(rhoa) // 3)
            smoothed = np.convolve(rhoa, np.ones(window) / window, mode="same")
            correlation = np.corrcoef(rhoa, smoothed)[0, 1]
            metrics["smoothness_score"] = (
                correlation if not np.isnan(correlation) else 1.0
            )
        else:
            metrics["smoothness_score"] = 1.0

        return metrics

    def _analyze_coverage_resolution(self, ab2) -> Dict[str, float]:
        """Analyze measurement coverage and resolution"""
        metrics = {}

        # Data span
        metrics["data_span"] = np.log10(ab2.max() / ab2.min())

        # Measurement density
        metrics["measurement_density"] = (
            len(ab2) / metrics["data_span"] if metrics["data_span"] > 0 else 0
        )

        # Resolution estimate (mean spacing in log domain)
        if len(ab2) > 1:
            log_ab2 = np.log10(ab2)
            spacings = np.diff(log_ab2)
            metrics["mean_log_spacing"] = np.mean(spacings)
            metrics["std_log_spacing"] = np.std(spacings)
        else:
            metrics["mean_log_spacing"] = 0
            metrics["std_log_spacing"] = 0

        return metrics

    def _detect_outliers(self, rhoa) -> np.ndarray:
        """Detect outliers using modified Z-score"""
        if len(rhoa) < 3:
            return np.array([])

        log_rhoa = np.log10(rhoa)

        # Modified Z-score (more robust than standard Z-score)
        median = np.median(log_rhoa)
        mad = np.median(np.abs(log_rhoa - median))

        if mad == 0:  # All values identical
            return np.array([])

        modified_z_scores = 0.6745 * (log_rhoa - median) / mad

        # Threshold for outliers (3.5 is commonly used)
        outlier_threshold = 3.5
        outliers = np.where(np.abs(modified_z_scores) > outlier_threshold)[0]

        return outliers

    def _generate_recommendations(
        self,
        ab2,
        rhoa,
        quality_metrics,
        noise_metrics,
        coverage_metrics,
        n_issues,
        n_warnings,
    ) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []

        # Data span recommendations
        if coverage_metrics["data_span"] < self.thresholds["min_data_span"]:
            recommendations.append(
                f"Increase AB2 range - current span ({coverage_metrics['data_span']:.1f} decades) "
                f"is below minimum ({self.thresholds['min_data_span']} decades)"
            )
        elif coverage_metrics["data_span"] > self.thresholds["ideal_data_span"]:
            recommendations.append("Excellent data span for deep investigation")

        # Measurement density
        if coverage_metrics["measurement_density"] < 3:
            recommendations.append(
                "Consider additional measurements for better resolution"
            )
        elif coverage_metrics["measurement_density"] > 10:
            recommendations.append("Good measurement density for detailed analysis")

        # Noise level recommendations
        if noise_metrics["noise_level"] > 0.1:
            recommendations.append(
                "High noise detected - consider data smoothing or remeasurement"
            )
        elif noise_metrics["smoothness_score"] < 0.8:
            recommendations.append("Irregular curve shape - verify measurement quality")

        # Overall quality recommendations
        if n_issues > 0:
            recommendations.append(
                "Address critical issues before proceeding with inversion"
            )
        if n_warnings > 3:
            recommendations.append(
                "Multiple data quality concerns - consider additional QC"
            )

        return recommendations

    def _calculate_quality_score(
        self, n_issues, n_warnings, quality_metrics, noise_metrics, coverage_metrics
    ) -> float:
        """Calculate overall quality score (0-100)"""
        base_score = 100.0

        # Penalize issues and warnings
        base_score -= n_issues * 25  # Heavy penalty for issues
        base_score -= n_warnings * 5  # Light penalty for warnings

        # Data span bonus/penalty
        data_span = coverage_metrics["data_span"]
        if data_span >= self.thresholds["ideal_data_span"]:
            base_score += 10  # Bonus for good span
        elif data_span < self.thresholds["min_data_span"]:
            base_score -= 15  # Penalty for poor span

        # Measurement density
        density = coverage_metrics["measurement_density"]
        if density >= 5:
            base_score += 5  # Bonus for good density
        elif density < 3:
            base_score -= 10  # Penalty for poor density

        # Noise penalty
        noise_level = noise_metrics["noise_level"]
        base_score -= noise_level * 100

        # Smoothness bonus
        smoothness = noise_metrics["smoothness_score"]
        base_score += (smoothness - 0.8) * 25

        return max(0, min(100, base_score))

    def _determine_quality_level(self, quality_score: float) -> DataQualityLevel:
        """Determine quality level from score"""
        if quality_score >= 90:
            return DataQualityLevel.EXCELLENT
        elif quality_score >= 75:
            return DataQualityLevel.GOOD
        elif quality_score >= 60:
            return DataQualityLevel.ACCEPTABLE
        elif quality_score >= 40:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNACCEPTABLE
