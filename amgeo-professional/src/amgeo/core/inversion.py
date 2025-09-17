# src/amgeo/core/inversion.py
"""
Professional VES inversion engine with multiple methods and uncertainty quantification
Combines the best features from PyGIMLi, fallback methods, and robust algorithms
Optimized for real-world groundwater exploration applications
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from enum import Enum
from scipy import optimize, stats
from scipy.interpolate import interp1d
import warnings

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Check for optional advanced libraries
PYGIMLI_AVAILABLE = False
SIMPEG_AVAILABLE = False

try:
    import pygimli as pg
    from pygimli.physics import ert

    PYGIMLI_AVAILABLE = True
    logger.info("PyGIMLi available for professional inversion")
except ImportError:
    logger.warning("PyGIMLi not available - using fallback methods")

try:
    import simpeg

    SIMPEG_AVAILABLE = True
    logger.info("SimPEG available for advanced inversion")
except ImportError:
    logger.warning("SimPEG not available - using fallback methods")


class InversionMethod(Enum):
    """Available inversion methods"""

    PYGIMLI = "pygimli"
    SIMPEG = "simpeg"
    MARQUARDT_LEVENBERG = "marquardt_levenberg"
    RIDGE_REGRESSION = "ridge_regression"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    OCCAM = "occam"
    ENSEMBLE = "ensemble"


class GeologicalContext(Enum):
    """Geological contexts for aquifer interpretation"""

    ALLUVIAL = "alluvial"
    SEDIMENTARY = "sedimentary"
    CRYSTALLINE = "crystalline"
    VOLCANIC = "volcanic"
    COASTAL = "coastal"
    KARST = "karst"
    WEATHERED = "weathered"
    FRACTURED = "fractured"


@dataclass
class InversionResult:
    """Comprehensive inversion result structure"""

    # Core results
    resistivities: np.ndarray
    thicknesses: np.ndarray
    depths: np.ndarray
    fitted_rhoa: np.ndarray

    # Input data
    ab2: np.ndarray
    mn2: np.ndarray
    rhoa: np.ndarray

    # Statistics
    chi2: float
    rms_error: float
    n_iterations: int

    # Method info
    method: str
    lambda_value: float

    # Advanced features
    resistivity_uncertainty: Optional[np.ndarray] = None
    thickness_uncertainty: Optional[np.ndarray] = None
    model_covariance: Optional[np.ndarray] = None
    resolution_matrix: Optional[np.ndarray] = None

    # Quality metrics
    data_fit_quality: str = "unknown"  # excellent, good, acceptable, poor
    model_roughness: float = 0.0
    convergence_achieved: bool = False

    # Aquifer analysis
    aquifer_layers: List[int] = None
    aquifer_probability: float = 0.0
    depth_to_aquifer: float = np.inf
    aquifer_thickness: float = 0.0

    # Metadata
    site_info: Dict[str, Any] = None
    processing_metadata: Dict[str, Any] = None


@dataclass
class AquiferAssessment:
    """Comprehensive aquifer assessment results"""

    probability: float  # 0-1 probability of aquifer presence
    confidence: float  # 0-1 confidence in assessment
    aquifer_layers: List[int]  # Layer indices identified as aquifers
    depth_range: Tuple[float, float]  # Depth range of aquifer zones
    total_thickness: float  # Total aquifer thickness
    average_resistivity: float  # Average resistivity of aquifer zones
    hydraulic_conductivity_estimate: float  # Estimated K (m/day)
    transmissivity_estimate: float  # Estimated T (m²/day)
    water_quality_indicator: str  # Fresh, Brackish, Saline
    aquifer_type: str  # Confined, Unconfined, Perched
    geological_interpretation: str
    recommendation: str


class VESInversionEngine:
    """
    Professional VES inversion engine with multiple methods and comprehensive analysis
    """

    # Aquifer resistivity ranges for different geological contexts (Ohm-m)
    AQUIFER_RESISTIVITY_RANGES = {
        GeologicalContext.ALLUVIAL: (20, 200),
        GeologicalContext.SEDIMENTARY: (30, 300),
        GeologicalContext.CRYSTALLINE: (100, 1000),
        GeologicalContext.VOLCANIC: (50, 500),
        GeologicalContext.COASTAL: (5, 50),
        GeologicalContext.KARST: (100, 2000),
        GeologicalContext.WEATHERED: (30, 200),
        GeologicalContext.FRACTURED: (200, 2000),
    }

    # Gosh filter coefficients for accurate forward modeling
    GOSH_FILTER = np.array(
        [
            6.4e-7,
            1.6e-6,
            4.0e-6,
            1.0e-5,
            2.5e-5,
            6.25e-5,
            1.56e-4,
            3.9e-4,
            9.77e-4,
            2.44e-3,
            6.1e-3,
            1.53e-2,
            3.81e-2,
            9.52e-2,
            0.238,
            0.595,
            1.484,
            3.7,
            9.23,
            23.0,
            57.4,
            143.0,
            357.0,
            890.0,
        ]
    )

    def __init__(self, settings=None, geological_context=GeologicalContext.ALLUVIAL):
        self.settings = settings or get_settings()
        self.inversion_config = self.settings.inversion
        self.geological_context = geological_context

        # Check available methods
        self.available_methods = self._detect_available_methods()
        logger.info(
            f"Available inversion methods: {list(self.available_methods.keys())}"
        )

    def _detect_available_methods(self) -> Dict[InversionMethod, bool]:
        """Detect which inversion methods are available"""
        methods = {
            InversionMethod.MARQUARDT_LEVENBERG: True,
            InversionMethod.RIDGE_REGRESSION: True,
            InversionMethod.DIFFERENTIAL_EVOLUTION: True,
            InversionMethod.ENSEMBLE: True,
        }

        if PYGIMLI_AVAILABLE:
            methods[InversionMethod.PYGIMLI] = True
        else:
            methods[InversionMethod.PYGIMLI] = False

        if SIMPEG_AVAILABLE:
            methods[InversionMethod.SIMPEG] = True
        else:
            methods[InversionMethod.SIMPEG] = False

        return methods

    def run_inversion(
        self,
        ab2: np.ndarray,
        mn2: np.ndarray,
        rhoa: np.ndarray,
        method: Union[str, InversionMethod] = None,
        n_layers: int = None,
        lambda_reg: float = None,
        max_iterations: int = None,
        error_level: float = None,
        verbose: bool = False,
        **kwargs,
    ) -> InversionResult:
        """
        Run comprehensive VES inversion with specified or automatic method selection

        Args:
            ab2: Half current electrode spacing array
            mn2: Half potential electrode spacing array
            rhoa: Apparent resistivity array
            method: Inversion method to use
            n_layers: Number of layers in model
            lambda_reg: Regularization parameter
            max_iterations: Maximum iterations
            error_level: Data error level
            verbose: Verbose output
            **kwargs: Additional method-specific parameters

        Returns:
            InversionResult with comprehensive results and aquifer assessment
        """

        # Set defaults from configuration
        method = method or self.inversion_config.default_method
        n_layers = n_layers or 4
        lambda_reg = lambda_reg or self.inversion_config.default_lambda
        max_iterations = max_iterations or self.inversion_config.max_iterations
        error_level = error_level or self.inversion_config.default_error_level

        # Convert string method to enum
        if isinstance(method, str):
            try:
                method = InversionMethod(method)
            except ValueError:
                logger.warning(f"Unknown method '{method}', using default")
                method = InversionMethod(self.inversion_config.default_method)

        # Validate input data
        if not self._validate_input_data(ab2, mn2, rhoa):
            raise ValueError(
                "Invalid input data - check electrode spacings and resistivity values"
            )

        # Preprocess data for better results
        ab2_proc, rhoa_proc = self._preprocess_data(ab2, rhoa)

        logger.info(
            f"Running {method.value} inversion with {n_layers} layers, λ={lambda_reg}"
        )

        # Run primary method with comprehensive error handling
        try:
            if method == InversionMethod.PYGIMLI and self.available_methods[method]:
                result = self._run_pygimli_inversion(
                    ab2_proc,
                    mn2,
                    rhoa_proc,
                    n_layers,
                    lambda_reg,
                    max_iterations,
                    error_level,
                    verbose,
                    **kwargs,
                )
            elif method == InversionMethod.SIMPEG and self.available_methods[method]:
                result = self._run_simpeg_inversion(
                    ab2_proc,
                    mn2,
                    rhoa_proc,
                    n_layers,
                    lambda_reg,
                    max_iterations,
                    error_level,
                    verbose,
                    **kwargs,
                )
            elif method == InversionMethod.ENSEMBLE:
                result = self._run_ensemble_inversion(
                    ab2_proc,
                    mn2,
                    rhoa_proc,
                    n_layers,
                    lambda_reg,
                    max_iterations,
                    error_level,
                    verbose,
                    **kwargs,
                )
            elif method == InversionMethod.MARQUARDT_LEVENBERG:
                result = self._run_marquardt_levenberg_inversion(
                    ab2_proc,
                    mn2,
                    rhoa_proc,
                    n_layers,
                    lambda_reg,
                    max_iterations,
                    error_level,
                    verbose,
                    **kwargs,
                )
            elif method == InversionMethod.DIFFERENTIAL_EVOLUTION:
                result = self._run_differential_evolution_inversion(
                    ab2_proc,
                    mn2,
                    rhoa_proc,
                    n_layers,
                    lambda_reg,
                    max_iterations,
                    error_level,
                    verbose,
                    **kwargs,
                )
            else:
                result = self._run_robust_fallback_inversion(
                    ab2_proc,
                    mn2,
                    rhoa_proc,
                    n_layers,
                    lambda_reg,
                    max_iterations,
                    error_level,
                    verbose,
                    **kwargs,
                )

            # Add aquifer analysis
            aquifer_assessment = self._assess_aquifer_potential(result)
            result.aquifer_layers = aquifer_assessment.aquifer_layers
            result.aquifer_probability = aquifer_assessment.probability
            result.depth_to_aquifer = (
                aquifer_assessment.depth_range[0]
                if aquifer_assessment.depth_range[0] != np.inf
                else np.inf
            )
            result.aquifer_thickness = aquifer_assessment.total_thickness

            # Add quality assessment
            result.data_fit_quality = self._assess_data_fit_quality(result.rms_error)
            result.model_roughness = self._calculate_model_roughness(
                result.resistivities
            )

            logger.info(
                f"Inversion completed: χ²={result.chi2:.4f}, RMS={result.rms_error:.2f}%, "
                f"Aquifer probability={result.aquifer_probability:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Primary inversion method {method.value} failed: {e}")

            # Try fallback methods in order of preference
            fallback_methods = [
                InversionMethod.DIFFERENTIAL_EVOLUTION,
                InversionMethod.MARQUARDT_LEVENBERG,
                InversionMethod.RIDGE_REGRESSION,
            ]

            for fallback_method in fallback_methods:
                if fallback_method != method and self.available_methods.get(
                    fallback_method, False
                ):
                    try:
                        logger.info(f"Trying fallback method: {fallback_method.value}")

                        if fallback_method == InversionMethod.DIFFERENTIAL_EVOLUTION:
                            result = self._run_differential_evolution_inversion(
                                ab2_proc,
                                mn2,
                                rhoa_proc,
                                n_layers,
                                lambda_reg,
                                max_iterations,
                                error_level,
                                verbose,
                                **kwargs,
                            )
                        elif fallback_method == InversionMethod.MARQUARDT_LEVENBERG:
                            result = self._run_marquardt_levenberg_inversion(
                                ab2_proc,
                                mn2,
                                rhoa_proc,
                                n_layers,
                                lambda_reg,
                                max_iterations,
                                error_level,
                                verbose,
                                **kwargs,
                            )
                        else:
                            result = self._run_robust_fallback_inversion(
                                ab2_proc,
                                mn2,
                                rhoa_proc,
                                n_layers,
                                lambda_reg,
                                max_iterations,
                                error_level,
                                verbose,
                                **kwargs,
                            )

                        result.method = (
                            f"{method.value}_fallback_{fallback_method.value}"
                        )
                        logger.info(
                            f"Fallback inversion successful: χ²={result.chi2:.4f}"
                        )

                        # Add aquifer analysis for fallback result
                        aquifer_assessment = self._assess_aquifer_potential(result)
                        result.aquifer_layers = aquifer_assessment.aquifer_layers
                        result.aquifer_probability = aquifer_assessment.probability
                        result.depth_to_aquifer = (
                            aquifer_assessment.depth_range[0]
                            if aquifer_assessment.depth_range[0] != np.inf
                            else np.inf
                        )
                        result.aquifer_thickness = aquifer_assessment.total_thickness

                        return result

                    except Exception as fallback_error:
                        logger.warning(
                            f"Fallback method {fallback_method.value} also failed: {fallback_error}"
                        )
                        continue

            # Final fallback - simple curve matching
            try:
                result = self._run_simple_curve_matching(
                    ab2_proc, mn2, rhoa_proc, n_layers
                )
                result.method = f"{method.value}_simple_fallback"
                logger.info(f"Simple fallback successful: RMS={result.rms_error:.2f}%")
                return result
            except Exception as final_error:
                logger.error(
                    f"All inversion methods failed. Final error: {final_error}"
                )
                raise RuntimeError(f"All inversion methods failed. Last error: {e}")

    def _validate_input_data(
        self, ab2: np.ndarray, mn2: np.ndarray, rhoa: np.ndarray
    ) -> bool:
        """Validate input VES data quality"""
        try:
            # Basic checks
            if len(ab2) != len(rhoa) or len(mn2) != len(rhoa):
                return False
            if len(ab2) < 3:  # Minimum points for any meaningful inversion
                return False
            if np.any(ab2 <= 0) or np.any(mn2 <= 0) or np.any(rhoa <= 0):
                return False
            if not np.all(np.isfinite(ab2)) or not np.all(np.isfinite(rhoa)):
                return False
            if not np.all(ab2[1:] >= ab2[:-1]):  # Check non-decreasing spacing
                return False

            # Check AB2/MN2 ratios
            ratios = ab2 / mn2
            if np.any(ratios < 1.5) or np.any(ratios > 100):
                logger.warning("Some AB2/MN2 ratios are outside recommended range")

            return True
        except:
            return False

    def _preprocess_data(
        self, ab2: np.ndarray, rhoa: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess VES data for better inversion results"""
        # Remove outliers using modified Z-score
        log_rhoa = np.log10(rhoa)
        median = np.median(log_rhoa)
        mad = np.median(np.abs(log_rhoa - median))

        if mad > 0:
            modified_z_scores = 0.6745 * (log_rhoa - median) / mad
            # Keep data points with modified Z-score < 3.5
            mask = np.abs(modified_z_scores) < 3.5
            ab2_clean = ab2[mask]
            rhoa_clean = rhoa[mask]
        else:
            ab2_clean, rhoa_clean = ab2, rhoa

        # Apply mild smoothing if data is noisy
        if len(ab2_clean) > 8:
            # Calculate noise level
            if len(rhoa_clean) > 2:
                second_derivative = np.diff(np.log10(rhoa_clean), n=2)
                noise_level = np.std(second_derivative)

                if noise_level > 0.1:  # High noise threshold
                    window = min(3, len(ab2_clean) // 4)
                    if window >= 2:
                        rhoa_smooth = np.convolve(
                            rhoa_clean, np.ones(window) / window, mode="same"
                        )
                        return ab2_clean, rhoa_smooth

        return ab2_clean, rhoa_clean

    def _run_pygimli_inversion(
        self,
        ab2,
        mn2,
        rhoa,
        n_layers,
        lambda_reg,
        max_iterations,
        error_level,
        verbose,
        **kwargs,
    ):
        """PyGIMLi-based professional inversion"""
        try:
            # Create VES manager
            ves = ert.VESManager(verbose=verbose)

            # Set data with proper error handling
            error_abs = rhoa * error_level
            ves.setData(ab2, mn2, rhoa, error_abs)

            # Configure layer constraints
            ves.setLayerLimits(n_layers - 1)

            # Set bounds if provided
            if "resistivity_bounds" in kwargs:
                res_min, res_max = kwargs["resistivity_bounds"]
                ves.setParaLimits(res_min, res_max)

            # Run inversion
            model = ves.invert(lam=lambda_reg, maxIter=max_iterations, verbose=verbose)

            # Extract results with robust parameter extraction
            resistivities, thicknesses = self._extract_pygimli_parameters(
                model, n_layers
            )

            # Calculate depths
            depths = np.concatenate([[0], np.cumsum(thicknesses[:-1])])
            if len(depths) < len(resistivities):
                depths = np.concatenate([depths, [depths[-1] + 100]])

            # Get fitted data
            fitted_rhoa = ves.simulate(model)

            # Calculate statistics
            residuals = (rhoa - fitted_rhoa) / error_abs
            chi2 = np.mean(residuals**2)
            rms_error = np.sqrt(np.mean(((rhoa - fitted_rhoa) / rhoa) ** 2)) * 100

            # Get number of iterations
            n_iter = getattr(ves, "iter", max_iterations)

            return InversionResult(
                resistivities=resistivities,
                thicknesses=thicknesses,
                depths=depths,
                fitted_rhoa=fitted_rhoa,
                ab2=ab2,
                mn2=mn2,
                rhoa=rhoa,
                chi2=chi2,
                rms_error=rms_error,
                n_iterations=n_iter,
                method="pygimli",
                lambda_value=lambda_reg,
                convergence_achieved=chi2 < 2.0,
                processing_metadata={
                    "pygimli_version": pg.__version__,
                    "error_level": error_level,
                    "n_data_points": len(ab2),
                },
            )

        except Exception as e:
            logger.error(f"PyGIMLi inversion failed: {e}")
            raise

    def _extract_pygimli_parameters(self, model, n_layers):
        """Robustly extract parameters from PyGIMLi model"""
        model_array = np.array(model)
        n_params = len(model_array)

        if n_params == 2 * n_layers - 1:
            # Standard alternating format: res1, thick1, res2, thick2, ..., resN
            resistivities = model_array[::2]
            thicknesses = np.append(model_array[1::2], np.inf)
        elif n_params == n_layers:
            # Only resistivities provided
            resistivities = model_array
            # Generate reasonable thicknesses
            max_depth = 100  # Default investigation depth
            thicknesses = np.full(n_layers, max_depth / n_layers)
            thicknesses[-1] = np.inf
        else:
            # Handle other cases
            if n_params >= n_layers:
                resistivities = model_array[:n_layers]
                remaining = model_array[n_layers:]
                if len(remaining) >= n_layers - 1:
                    thicknesses = np.append(remaining[: n_layers - 1], np.inf)
                else:
                    # Generate missing thicknesses
                    max_depth = 100
                    thick_template = np.full(n_layers - 1, max_depth / n_layers)
                    thicknesses = np.append(thick_template, np.inf)
            else:
                # Too few parameters - generate reasonable defaults
                resistivities = np.full(
                    n_layers, np.median(model_array) if len(model_array) > 0 else 100
                )
                thicknesses = np.full(n_layers, 100 / n_layers)
                thicknesses[-1] = np.inf

        return np.asarray(resistivities), np.asarray(thicknesses)

    def _run_marquardt_levenberg_inversion(
        self,
        ab2,
        mn2,
        rhoa,
        n_layers,
        lambda_reg,
        max_iterations,
        error_level,
        verbose,
        **kwargs,
    ):
        """Marquardt-Levenberg (damped least squares) inversion"""

        def residuals_function(params):
            try:
                resistivities, thicknesses = self._split_parameters(params, n_layers)
                rhoa_calc = self._forward_model_schlumberger(
                    ab2, mn2, resistivities, thicknesses
                )
                residuals = (rhoa - rhoa_calc) / (rhoa * error_level)
                return residuals
            except:
                return np.full_like(rhoa, 1e6)  # Large residual for invalid parameters

        # Initial parameters with geological constraints
        initial_params = self._generate_initial_parameters(ab2, rhoa, n_layers)

        # Parameter bounds
        bounds_lower, bounds_upper = self._generate_parameter_bounds(n_layers, ab2)

        # Run Levenberg-Marquardt optimization
        try:
            result = optimize.least_squares(
                residuals_function,
                initial_params,
                bounds=(bounds_lower, bounds_upper),
                method="lm",
                max_nfev=max_iterations * 10,
                ftol=1e-8,
                xtol=1e-8,
            )

            # Extract results
            resistivities, thicknesses = self._split_parameters(result.x, n_layers)

            # Calculate fitted data and statistics
            fitted_rhoa = self._forward_model_schlumberger(
                ab2, mn2, resistivities, thicknesses
            )
            rms_error = np.sqrt(np.mean(((rhoa - fitted_rhoa) / rhoa) ** 2)) * 100
            chi2 = np.mean(((rhoa - fitted_rhoa) / (rhoa * error_level)) ** 2)

            # Calculate depths
            depths = np.concatenate([[0], np.cumsum(thicknesses[:-1])])

            return InversionResult(
                resistivities=resistivities,
                thicknesses=thicknesses,
                depths=depths,
                fitted_rhoa=fitted_rhoa,
                ab2=ab2,
                mn2=mn2,
                rhoa=rhoa,
                chi2=chi2,
                rms_error=rms_error,
                n_iterations=result.nfev,
                method="marquardt_levenberg",
                lambda_value=lambda_reg,
                convergence_achieved=result.success,
                processing_metadata={
                    "optimization_success": result.success,
                    "optimization_message": result.message,
                    "cost": result.cost,
                },
            )

        except Exception as e:
            logger.error(f"Marquardt-Levenberg inversion failed: {e}")
            raise

    def _run_differential_evolution_inversion(
        self,
        ab2,
        mn2,
        rhoa,
        n_layers,
        lambda_reg,
        max_iterations,
        error_level,
        verbose,
        **kwargs,
    ):
        """Robust differential evolution global optimization"""

        def objective_function(params):
            try:
                resistivities, thicknesses = self._split_parameters(params, n_layers)
                rhoa_calc = self._forward_model_schlumberger(
                    ab2, mn2, resistivities, thicknesses
                )

                # Use robust Huber loss
                residuals = (rhoa - rhoa_calc) / (rhoa * error_level)
                delta = 1.35  # Huber threshold

                huber_loss = np.where(
                    np.abs(residuals) <= delta,
                    0.5 * residuals**2,
                    delta * (np.abs(residuals) - 0.5 * delta),
                )

                # Add regularization term
                if n_layers > 1:
                    smoothness = np.sum(np.diff(np.log10(resistivities)) ** 2)
                    regularization = lambda_reg * smoothness / len(resistivities)
                else:
                    regularization = 0

                return np.sum(huber_loss) + regularization

            except:
                return 1e10  # Large penalty for invalid parameters

        # Parameter bounds
        bounds_lower, bounds_upper = self._generate_parameter_bounds(n_layers, ab2)
        bounds = list(zip(bounds_lower, bounds_upper))

        # Run differential evolution with multiple strategies
        strategies = ["best1bin", "best2bin", "rand1bin", "currenttobest1bin"]
        best_result = None
        best_cost = np.inf

        for strategy in strategies:
            try:
                result = optimize.differential_evolution(
                    objective_function,
                    bounds,
                    strategy=strategy,
                    maxiter=max_iterations,
                    popsize=15,
                    seed=42,
                    atol=1e-8,
                    tol=1e-8,
                    workers=1,  # Keep deterministic
                )

                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result

            except Exception as e:
                logger.warning(
                    f"Differential evolution strategy {strategy} failed: {e}"
                )
                continue

        if best_result is None:
            raise RuntimeError("All differential evolution strategies failed")

        # Extract results
        resistivities, thicknesses = self._split_parameters(best_result.x, n_layers)

        # Calculate fitted data and statistics
        fitted_rhoa = self._forward_model_schlumberger(
            ab2, mn2, resistivities, thicknesses
        )
        rms_error = np.sqrt(np.mean(((rhoa - fitted_rhoa) / rhoa) ** 2)) * 100
        chi2 = np.mean(((rhoa - fitted_rhoa) / (rhoa * error_level)) ** 2)

        # Calculate depths
        depths = np.concatenate([[0], np.cumsum(thicknesses[:-1])])

        return InversionResult(
            resistivities=resistivities,
            thicknesses=thicknesses,
            depths=depths,
            fitted_rhoa=fitted_rhoa,
            ab2=ab2,
            mn2=mn2,
            rhoa=rhoa,
            chi2=chi2,
            rms_error=rms_error,
            n_iterations=best_result.nit,
            method="differential_evolution",
            lambda_value=lambda_reg,
            convergence_achieved=best_result.success,
            processing_metadata={
                "optimization_success": best_result.success,
                "final_cost": best_result.fun,
                "message": best_result.message,
            },
        )

    def _run_ensemble_inversion(
        self,
        ab2,
        mn2,
        rhoa,
        n_layers,
        lambda_reg,
        max_iterations,
        error_level,
        verbose,
        **kwargs,
    ):
        """Ensemble inversion combining multiple methods"""

        logger.info("Running ensemble inversion with multiple methods")

        # Define methods and their weights
        methods_to_try = []
        weights = []

        if self.available_methods.get(InversionMethod.PYGIMLI, False):
            methods_to_try.append(("pygimli", self._run_pygimli_inversion))
            weights.append(1.3)

        methods_to_try.append(
            ("marquardt_levenberg", self._run_marquardt_levenberg_inversion)
        )
        weights.append(1.2)

        methods_to_try.append(
            ("differential_evolution", self._run_differential_evolution_inversion)
        )
        weights.append(1.1)

        # Run multiple inversions
        results = []
        successful_weights = []

        for method_name, method_func in methods_to_try:
            try:
                result = method_func(
                    ab2,
                    mn2,
                    rhoa,
                    n_layers,
                    lambda_reg,
                    max_iterations,
                    error_level,
                    False,
                    **kwargs,
                )
                results.append(result)
                successful_weights.append(weights[len(results) - 1])

                if verbose:
                    logger.info(
                        f"Ensemble member {method_name}: χ²={result.chi2:.3f}, RMS={result.rms_error:.2f}%"
                    )

            except Exception as e:
                logger.warning(f"Ensemble method {method_name} failed: {e}")
                continue

        if not results:
            raise RuntimeError("All ensemble methods failed")

        # Combine results using weighted averaging
        ensemble_result = self._combine_ensemble_results(results)
