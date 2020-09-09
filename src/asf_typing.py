"""
    Define custom type aliases here.
"""
from typing import Dict, List, Tuple

History = Dict[str, List[float]]
MaskedDatasetMetadata = List[Tuple[str, str, str]]

"""Subdataset name, Timeseries path key (ulx_###_uly_####)"""
TimeseriesMetadataFrameKey = Tuple[str, str]