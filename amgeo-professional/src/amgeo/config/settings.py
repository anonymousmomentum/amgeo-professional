# src/amgeo/config/settings.py

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import yaml
import logging
from enum import Enum

# Handle Pydantic v2 compatibility
try:
    # Try new pydantic-settings package first (Pydantic v2)
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        # Fallback to pydantic BaseSettings (Pydantic v1)
        from pydantic import BaseSettings
    except ImportError:
        # Create minimal fallback for extreme cases
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)


from pydantic import Field

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """Application environments"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """Database configuration with enhanced features"""

    url: str = Field(
        default="sqlite:///amdigits.db", description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Echo SQL queries for debugging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum connection overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")

    # Connection retry settings
    max_retries: int = Field(default=3, description="Maximum connection retries")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")

    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and sessions"""

    url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    decode_responses: bool = Field(default=True, description="Decode Redis responses")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(default=5.0, description="Connection timeout")
    max_connections: int = Field(default=10, description="Maximum connections in pool")

    # Cache settings
    default_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    key_prefix: str = Field(default="amgeo:", description="Redis key prefix")

    class Config:
        env_prefix = "REDIS_"


class InversionSettings(BaseSettings):
    """Enhanced VES Inversion configuration"""

    default_method: str = Field(
        default="pygimli", description="Default inversion method"
    )
    fallback_methods: List[str] = Field(
        default=["damped_lsq", "ensemble"],
        description="Fallback methods if primary fails",
    )
    max_iterations: int = Field(default=50, description="Maximum inversion iterations")
    default_lambda: float = Field(
        default=20.0, description="Default regularization parameter"
    )
    default_error_level: float = Field(
        default=0.03, description="Default data error level"
    )
    cache_results: bool = Field(default=True, description="Cache inversion results")

    # Quality control thresholds
    min_data_points: int = Field(default=5, description="Minimum required data points")
    max_rms_error: float = Field(
        default=15.0, description="Maximum acceptable RMS error"
    )
    min_ab2_spacing: float = Field(default=0.1, description="Minimum AB/2 spacing")
    max_ab2_spacing: float = Field(default=1000.0, description="Maximum AB/2 spacing")
    min_data_span_decades: float = Field(
        default=1.5, description="Minimum data span in log decades"
    )

    # Advanced parameters
    convergence_tolerance: float = Field(
        default=1e-6, description="Convergence tolerance"
    )
    max_ab2_mn2_ratio: float = Field(default=10.0, description="Maximum AB2/MN2 ratio")
    outlier_threshold: float = Field(
        default=3.0, description="Outlier detection threshold (std dev)"
    )
    auto_fix_enabled: bool = Field(
        default=True, description="Enable automatic data fixing"
    )

    # Uncertainty quantification
    bootstrap_samples: int = Field(
        default=100, description="Bootstrap samples for uncertainty"
    )
    confidence_level: float = Field(
        default=0.95, description="Confidence level for intervals"
    )

    class Config:
        env_prefix = "INVERSION_"


class MLSettings(BaseSettings):
    """Enhanced Machine Learning configuration"""

    model_cache_dir: str = Field(
        default="data/models", description="Model cache directory"
    )
    feature_cache_dir: str = Field(
        default="data/processed", description="Feature cache directory"
    )
    default_cv_folds: int = Field(default=5, description="Cross-validation folds")
    random_state: int = Field(default=42, description="Random seed for reproducibility")

    # Ensemble settings
    ensemble_methods: List[str] = Field(
        default=["random_forest", "gradient_boost", "svm_rbf", "neural_network"],
        description="ML methods in ensemble",
    )
    n_estimators: int = Field(default=200, description="Number of estimators")
    max_depth: Optional[int] = Field(default=None, description="Maximum tree depth")
    learning_rate: float = Field(default=0.1, description="Learning rate for boosting")

    # Feature engineering
    max_layers: int = Field(
        default=8, description="Maximum number of layers to extract"
    )
    feature_scaling: bool = Field(default=True, description="Enable feature scaling")
    feature_selection: bool = Field(
        default=True, description="Enable feature selection"
    )
    pca_components: Optional[int] = Field(
        default=None, description="PCA components (None = no PCA)"
    )

    # Training settings
    test_size: float = Field(default=0.2, description="Test set proportion")
    validation_size: float = Field(default=0.2, description="Validation set proportion")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, description="Early stopping patience")

    # Uncertainty quantification
    uncertainty_samples: int = Field(
        default=100, description="Samples for uncertainty estimation"
    )
    confidence_intervals: bool = Field(
        default=True, description="Calculate confidence intervals"
    )

    class Config:
        env_prefix = "ML_"


class APISettings(BaseSettings):
    """API server configuration"""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    workers: int = Field(default=1, description="Number of worker processes")

    # Security settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Request settings
    max_request_size: int = Field(
        default=16777216, description="Maximum request size in bytes"
    )  # 16MB
    request_timeout: int = Field(default=30, description="Request timeout in seconds")

    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Enhanced logging configuration"""

    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Date format")

    # File logging
    file: Optional[str] = Field(default=None, description="Log file path")
    max_size: str = Field(default="10MB", description="Maximum log file size")
    backup_count: int = Field(default=5, description="Number of backup log files")

    # Advanced settings
    json_logs: bool = Field(default=False, description="Enable JSON formatted logs")
    correlation_id: bool = Field(default=True, description="Include correlation IDs")
    performance_logging: bool = Field(
        default=False, description="Enable performance logging"
    )

    # Component-specific logging
    sql_logging: bool = Field(default=False, description="Enable SQL query logging")
    http_logging: bool = Field(default=True, description="Enable HTTP request logging")

    class Config:
        env_prefix = "LOGGING_"


class SecuritySettings(BaseSettings):
    """Security configuration"""

    secret_key: str = Field(
        default="dev-secret-key-change-in-production", description="Secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration"
    )
    refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration"
    )

    # Password settings
    password_min_length: int = Field(default=8, description="Minimum password length")
    password_require_special: bool = Field(
        default=True, description="Require special characters"
    )

    # Session settings
    session_timeout: int = Field(default=1800, description="Session timeout in seconds")
    max_login_attempts: int = Field(default=5, description="Maximum login attempts")
    lockout_duration: int = Field(
        default=300, description="Lockout duration in seconds"
    )

    class Config:
        env_prefix = "SECURITY_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""

    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    health_check_interval: int = Field(default=30, description="Health check interval")

    # Performance monitoring
    slow_query_threshold: float = Field(
        default=1.0, description="Slow query threshold in seconds"
    )
    memory_threshold: float = Field(default=0.8, description="Memory usage threshold")
    disk_threshold: float = Field(default=0.9, description="Disk usage threshold")

    # External monitoring
    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error tracking"
    )
    prometheus_enabled: bool = Field(
        default=False, description="Enable Prometheus metrics"
    )
    jaeger_endpoint: Optional[str] = Field(
        default=None, description="Jaeger tracing endpoint"
    )

    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings with comprehensive configuration"""

    # Application metadata
    app_name: str = Field(default="AMgeo Professional", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    description: str = Field(
        default="Advanced Groundwater Exploration Suite",
        description="Application description",
    )

    # Environment settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    testing: bool = Field(default=False, description="Enable testing mode")

    # Component configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    inversion: InversionSettings = Field(default_factory=InversionSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # Directory structure
    base_dir: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = base_dir / "data"
    exports_dir: Path = base_dir / "exports"
    logs_dir: Path = base_dir / "logs"
    temp_dir: Path = base_dir / "temp"
    config_dir: Path = base_dir / "config"
    models_dir: Path = base_dir / "models"
    uploads_dir: Path = base_dir / "uploads"

    # External integrations
    enable_telemetry: bool = Field(default=False, description="Enable usage telemetry")
    update_check: bool = Field(default=True, description="Check for updates")
    analytics_enabled: bool = Field(default=False, description="Enable analytics")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        # Pydantic v2 compatibility
        if hasattr(BaseSettings, "model_config"):
            model_config = {
                "env_file": ".env",
                "env_file_encoding": "utf-8",
                "case_sensitive": False,
                "extra": "ignore",
            }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._post_init()

    def _post_init(self):
        """Post-initialization setup"""
        self.create_directories()
        self._setup_logging()
        self._validate_configuration()

    @classmethod
    def load_from_yaml(cls, config_path: Path) -> "Settings":
        """Load settings from YAML file with error handling"""
        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
                return cls(**config_data)
            else:
                logger.warning(f"Configuration file not found: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")

        return cls()

    def create_directories(self) -> None:
        """Create all necessary directories"""
        directories = [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "models",
            self.data_dir / "cache",
            self.exports_dir,
            self.logs_dir,
            self.temp_dir,
            self.models_dir,
            self.uploads_dir,
            Path(self.ml.model_cache_dir),
            Path(self.ml.feature_cache_dir),
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")

    def _setup_logging(self):
        """Configure application logging"""
        try:
            # Configure root logger
            log_format = self.logging.format
            log_level = getattr(logging, self.logging.level.value)

            handlers = [logging.StreamHandler()]

            # Add file handler if specified
            if self.logging.file:
                log_file = Path(self.logging.file)
                log_file.parent.mkdir(parents=True, exist_ok=True)

                if self.logging.backup_count > 0:
                    from logging.handlers import RotatingFileHandler

                    file_handler = RotatingFileHandler(
                        log_file,
                        maxBytes=self._parse_size(self.logging.max_size),
                        backupCount=self.logging.backup_count,
                    )
                else:
                    file_handler = logging.FileHandler(log_file)

                handlers.append(file_handler)

            # Configure logging
            logging.basicConfig(
                level=log_level,
                format=log_format,
                datefmt=self.logging.date_format,
                handlers=handlers,
                force=True,
            )

            # Set specific logger levels
            if self.debug:
                logging.getLogger("amgeo").setLevel(logging.DEBUG)

            if not self.logging.sql_logging:
                logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

        except Exception as e:
            print(f"Failed to setup logging: {e}")

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes"""
        size_str = size_str.upper()
        multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                return int(size_str[: -len(suffix)]) * multiplier

        return int(size_str)  # Assume bytes if no suffix

    def _validate_configuration(self):
        """Validate configuration settings"""
        warnings = []

        # Validate directories
        if not self.base_dir.exists():
            warnings.append(f"Base directory does not exist: {self.base_dir}")

        # Validate security settings
        if self.environment == Environment.PRODUCTION:
            if self.security.secret_key == "dev-secret-key-change-in-production":
                warnings.append("Using default secret key in production")
            if self.debug:
                warnings.append("Debug mode is enabled in production")

        # Validate inversion settings
        if self.inversion.default_lambda <= 0:
            warnings.append("Invalid regularization parameter")

        # Validate ML settings
        if self.ml.test_size + self.ml.validation_size >= 1.0:
            warnings.append("Test and validation sizes sum to >= 1.0")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

    def get_database_url(self) -> str:
        """Get formatted database URL"""
        return str(self.database.url)

    def get_model_path(self, model_name: str) -> Path:
        """Get path for a specific model"""
        return Path(self.ml.model_cache_dir) / f"{model_name}.pkl"

    def get_output_path(self, filename: str) -> Path:
        """Get path for output file"""
        return self.exports_dir / filename

    def get_temp_path(self, filename: str) -> Path:
        """Get path for temporary file"""
        return self.temp_dir / filename

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING or self.testing

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        try:
            if hasattr(self, "model_dump"):
                return self.model_dump()  # Pydantic v2
            else:
                return self.dict()  # Pydantic v1
        except Exception:
            # Fallback for custom implementation
            return {
                key: getattr(self, key)
                for key in dir(self)
                if not key.startswith("_") and not callable(getattr(self, key))
            }


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get singleton settings instance with comprehensive error handling"""
    global _settings_instance

    if _settings_instance is None:
        try:
            # Try to load from environment-specific config
            env = os.getenv("ENVIRONMENT", "development")
            config_path = Path(f"config/{env}.yaml")

            if config_path.exists():
                _settings_instance = Settings.load_from_yaml(config_path)
            else:
                _settings_instance = Settings()

            logger.info(f"Settings loaded successfully for environment: {env}")

        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            _settings_instance = _create_fallback_settings()

    return _settings_instance


def _create_fallback_settings() -> Settings:
    """Create minimal fallback settings for emergency situations"""
    logger.warning("Using minimal fallback settings due to configuration errors")

    try:
        return Settings(
            app_name="AMgeo Professional",
            version="0.1.0",
            environment=Environment.DEVELOPMENT,
            debug=True,
        )
    except Exception as e:
        logger.error(f"Even fallback settings failed: {e}")

        # Create most basic fallback
        class MinimalSettings:
            app_name = "AMgeo Professional"
            version = "0.1.0"
            debug = True
            environment = "development"

            class inversion:
                default_method = "damped_lsq"
                default_lambda = 20.0
                max_iterations = 50
                default_error_level = 0.03
                fallback_methods = ["damped_lsq"]
                min_data_points = 5
                cache_results = False
                min_ab2_spacing = 0.1
                max_ab2_spacing = 1000.0

            class ml:
                model_cache_dir = Path("models")
                feature_cache_enabled = False
                uncertainty_samples = 50
                ensemble_methods = ["random_forest"]
                random_state = 42

            class database:
                url = "sqlite:///amdigits.db"
                echo = False

            class logging:
                level = LogLevel.INFO
                file = None

            def is_development(self):
                return True

            def is_production(self):
                return False

            def create_directories(self):
                pass

        return MinimalSettings()


def reload_settings():
    """Reload settings (useful for testing or configuration changes)"""
    global _settings_instance
    _settings_instance = None
    return get_settings()


def update_settings(**kwargs) -> Settings:
    """Update settings with new values"""
    global _settings_instance
    if _settings_instance is not None:
        for key, value in kwargs.items():
            if hasattr(_settings_instance, key):
                setattr(_settings_instance, key, value)
    return get_settings()


# Environment-specific configurations
def get_development_overrides() -> Dict[str, Any]:
    """Get development environment overrides"""
    return {
        "debug": True,
        "logging": {"level": LogLevel.DEBUG},
        "database": {"echo": True},
        "monitoring": {"enabled": False},
        "security": {"secret_key": "dev-secret-key"},
    }


def get_production_overrides() -> Dict[str, Any]:
    """Get production environment overrides"""
    return {
        "debug": False,
        "logging": {"level": LogLevel.INFO, "file": "logs/amgeo.log"},
        "database": {"echo": False, "pool_size": 20},
        "api": {"workers": 4, "reload": False},
        "monitoring": {"enabled": True, "prometheus_enabled": True},
    }


def get_testing_overrides() -> Dict[str, Any]:
    """Get testing environment overrides"""
    return {
        "debug": True,
        "testing": True,
        "logging": {"level": LogLevel.WARNING},
        "database": {"url": "sqlite:///:memory:", "echo": False},
        "ml": {"uncertainty_samples": 10},  # Faster for tests
        "inversion": {"cache_results": False},
    }
