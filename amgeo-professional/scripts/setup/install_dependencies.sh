#!/bin/bash
# Install additional geophysical libraries

set -e

echo "Installing PyGIMLi and other geophysical tools..."

# Install pygimli and other libraries from conda-forge (more reliable than pip)
if command -v conda &> /dev/null; then
    poetry run conda install -c conda-forge pygimli resipy simpeg -y
else
    echo "Warning: conda not found. Attempting pip install of geophysical libraries..."
    poetry run pip install \
        pygimli \
        resipy \
        emg3d \
        simpeg
fi

echo "Geophysical libraries installed successfully"