"""types."""
from typing import Union

import numpy as np
import pandas as pd


try:
    from cudf import DataFrame as CDataFrame
    from cudf import Series as CSeries
    from cupy import ndarray as CNDArray
except ImportError:
    CDataFrame = None
    CSeries = None
    CNDArray = None

XDataFrame = Union[CDataFrame, pd.DataFrame]
XSeries = Union[CSeries, np.ndarray]
XSeriesB = Union[CSeries, pd.Series]
XNDArray = Union[CNDArray, np.ndarray]
