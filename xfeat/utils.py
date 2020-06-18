from typing import List, Tuple
from logging import getLogger

import pandas as pd
import numpy as np

from xfeat.types import XDataFrame


logger = getLogger(__name__)


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


def analyze_columns(input_df: XDataFrame) -> Tuple[List[str], List[str]]:
    """Classify columns to numerical or categorical.

    Args:
        input_df (XDataFrame) : Input data frame.
    Returns:
        Tuple[List[str], List[str]] : List of num cols and cat cols.

    Example:
        ::
            >>> import pandas as pd
            >>> from xfeat.utils import analyze_columns
            >>> df = pd.DataFrame({"col1": [1, 2], "col2": [2, 3], "col3": ["a", "b"]})
            >>> analyze_columns(df)
            (['col1', 'col2'], ['col3'])
    """
    numerical_cols = []
    categorical_cols = []
    for col in input_df.columns:
        if pd.api.types.is_numeric_dtype(input_df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    return numerical_cols, categorical_cols


def compress_df(df: XDataFrame, verbose=False) -> XDataFrame:
    """Reduce memory usage by converting data types.

    For compatibility with feather, float16 is not used.

    Returns:
        The reduce data frame.
    """
    _num_dtypes = [
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    ]
    start_mem_usage = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type in _num_dtypes:
            min_val, max_val = df[col].min(), df[col].max()
            if str(col_type).startswith("int"):
                if (
                    min_val >= np.iinfo(np.int8).min
                    and max_val <= np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    min_val >= np.iinfo(np.int16).min
                    and max_val <= np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    min_val >= np.iinfo(np.int32).min
                    and max_val <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    min_val >= np.iinfo(np.int64).min
                    and max_val <= np.iinfo(np.int64).max
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                # NOTE: half float is not supported in feather.

                if (
                    min_val >= np.finfo(np.float32).min
                    and max_val <= np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem_usage = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.warning(
            "Memory reduced from {:.2f} MB to {:.2f} MB".format(
                start_mem_usage, end_mem_usage,
            )
        )

    return df


def cudf_is_available() -> bool:
    """Check avilability of CuDF.

    Returns:
        Available or not.
    """
    return cudf is not None


def is_cudf(input_df: XDataFrame) -> bool:
    """Check whether the input dataframe is cudf or not.

    Returns:
        cuDF or not.
    """
    return cudf_is_available() and isinstance(input_df, cudf.DataFrame)
