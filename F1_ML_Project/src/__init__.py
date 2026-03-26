"""
F1 Machine Learning Analytics Package
======================================
A comprehensive ML framework for Formula 1 race prediction,
hypothetical scenario modeling, and GOAT analysis.

Modules:
    data_loader         - Load and merge all 14 F1 CSV files
    feature_engineering - Compute 39 ML features with no data leakage
    models              - Train, tune, and predict with XGBoost/LightGBM/RF
    hypothetical_engine - 20-parameter hypothetical scenario engine
    goat_analysis       - 5-lens GOAT scoring framework
    visualizations      - Publication-quality F1 visualizations
"""

from . import data_loader

from . import feature_engineering
from . import models

from . import hypothetical_engine

# The modules below are imported lazily to avoid ImportError while they are
# still under development.  Uncomment each line once the corresponding file
# has been added to this package.
# from . import goat_analysis
# from . import visualizations

__version__ = "1.0.0"
__author__ = "F1 ML Analytics"
