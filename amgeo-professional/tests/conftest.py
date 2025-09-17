# tests/conftest.py
"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_ves_data():
    """Generate synthetic VES data for testing"""
    ab2 = np.logspace(-0.3, 2, 20)  # 0.5 to 100 m
    mn2 = ab2 / 3

    # Simple 3-layer model: 100-20-300 Ohm-m, thicknesses: 5-15 m
    resistivities = [100, 20, 300]
    thicknesses = [5, 15]

    # Simple forward modeling (approximation)
    rhoa = np.full_like(ab2, resistivities[0])  # Start with surface layer

    # Apply layer effects (simplified)
    for i, spacing in enumerate(ab2):
        inv_depth = spacing / 3
        if inv_depth > 5:
            rhoa[i] = resistivities[1]  # Conductor influence
        if inv_depth > 20:
            rhoa[i] = resistivities[2]  # Basement influence
