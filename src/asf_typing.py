"""
    Define custom type aliases here.
"""
from typing import Dict, List, Tuple

History = Dict[str, List[float]]
MaskedDatasetMetadata = List[Tuple[str, str, str]]
# MaskedTimeseriesMetadata = List[Tuple[List[Tuple[str, str]], str]]
MaskedTimeseriesMetadata = List[Tuple[List[Tuple[str, str]], str]]
"""Each time series data point has a list vh vv pairs, paired with a single mask"""