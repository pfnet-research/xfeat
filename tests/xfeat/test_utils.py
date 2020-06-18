import pytest
import numpy as np
import pandas as pd
from xfeat.utils import analyze_columns
from xfeat.utils import compress_df
from xfeat.utils import cudf_is_available

import pandas as pd


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"col": ["A", "B", "B"], "num": [1, 2, 3],})

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_compress_df(dataframes):
    for df in dataframes:
        assert str(df["num"].dtype) == "int64"
        df_compressed = compress_df(df)
        assert str(df_compressed["num"].dtype) == "int8"


def test_cudf_is_available():
    if cudf is None:
        assert cudf_is_available() is False
    else:
        assert cudf_is_available() is True


def test_analyze_columns(dataframes):
    for df in dataframes:
        num_cols, cat_cols = analyze_columns(df)
        assert num_cols == ["num"]
        assert cat_cols == ["col"]
