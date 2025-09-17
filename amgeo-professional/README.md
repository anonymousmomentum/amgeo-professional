# AMgeo Professional - Advanced Groundwater Exploration Suite

A comprehensive, industry-standard groundwater exploration tool using VES (Vertical Electrical Sounding) data analysis and machine learning for aquifer detection.

## Features

- 🔬 **Professional VES Inversion**: PyGIMLi integration with multiple fallback methods
- 🤖 **Advanced ML Pipeline**: Ensemble learning with uncertainty quantification
- 📊 **Industry-Standard Reporting**: ASTM D6431 compliant analysis and documentation
- 🗺️ **Spatial Analysis**: Multi-site groundwater mapping with GIS integration
- 🔍 **Quality Assurance**: Comprehensive validation and QA/QC framework
- 💾 **Enterprise Database**: PostgreSQL with PostGIS for spatial data management

## Quick Start

### Prerequisites

- Linux (Manjaro/Arch recommended)
- Python 3.9+
- Poetry
- Docker & Docker Compose
- PostgreSQL with PostGIS

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd amgeo-professional
   ./scripts/dev.sh install
   ```

2. **Start development services:**
   ```bash
   ./scripts/dev.sh start
   ```

3. **Access the application:**
   - Streamlit App: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Jupyter Lab: http://localhost:8888

### Development Workflow

- **Start development server**: `./scripts/dev.sh start`
- **Run tests**: `./scripts/dev.sh test`
- **Code formatting**: `./scripts/dev.sh lint`
- **Full Docker environment**: `./scripts/dev.sh docker`

## Architecture

```
src/amgeo/
├── core/           # Core VES analysis and inversion
├── ml/             # Machine learning pipeline
├── database/       # Database models and operations
├── api/            # REST API
├── visualization/  # Plotting and visualization
├── utils/          # Utility functions
└── config/         # Configuration management
```

## Documentation

- [User Guide](docs/user/README.md)
- [Developer Guide](docs/developer/README.md)
- [API Documentation](docs/api/README.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `./scripts/dev.sh lint`
5. Submit a pull request

## License

MIT License - see LICENSE file for details
