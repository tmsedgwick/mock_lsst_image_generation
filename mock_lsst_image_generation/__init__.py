"""Mock LSST-like galaxy catalogues (and, later, images) for training and testing source detectors."""

from .catalogue_pipeline import MockCatalogue, build_mock_catalogue, generate_mock_catalogues, save_mock_catalogue
from .config import CATALOGUE_STEM, CONFIG, N_CATALOGUES, PHYS, catalogue_splits
from .cosmos_catalogue import DEFAULT_COSMOS_PATH, load_cosmos2025_catalogue

__all__ = ["CATALOGUE_STEM", "CONFIG", "DEFAULT_COSMOS_PATH", "N_CATALOGUES", "PHYS", "MockCatalogue", "build_mock_catalogue",
           "catalogue_splits", "generate_mock_catalogues", "load_cosmos2025_catalogue", "save_mock_catalogue"]
