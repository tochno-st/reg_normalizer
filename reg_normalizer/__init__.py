"""
Region Normalizer - Tool for normalizing and standardizing Russian region names.

This package helps recognize regions even with typos, Latin characters, or various spelling variations.
It matches different forms of region names against an etalon reference and allows extracting additional
attributes such as OKATO codes, ISO codes, English names, and more.
"""

from .regions_validator import RegionMatcher

__version__ = "1.0.3"
__all__ = ["RegionMatcher"]
