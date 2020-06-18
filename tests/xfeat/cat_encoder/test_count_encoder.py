import pytest

import numpy as np
import pandas as pd

from xfeat import CountEncoder
from xfeat.cat_encoder._count_encoder import _CountEncoder
from xfeat.utils import cudf_is_available, is_cudf


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"col1": ["A", "B", "B", "C"],})

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


@pytest.fixture
def dataframes_num():
    df = pd.DataFrame(
        {
            "col1": [2, 2, 2, 8, 8, 8, 8],
            "col2": [2, 4, 6, 7, 8, 9, 10],
            "target": [1, 1, 0, 1, 1, 1, 0],
        }
    )

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_count_encoder_with_cat_cols(dataframes):
    for df in dataframes:
        encoder = CountEncoder()
        df_encoded = encoder.fit_transform(df)
        assert np.allclose(df_encoded["col1_ce"].values, np.array([1, 2, 2, 1]))
        assert df_encoded.columns.tolist() == [
            "col1",
            "col1_ce",
        ]


def test_count_encoder_with_num_cols(dataframes_num):
    for df in dataframes_num:
        encoder = CountEncoder(input_cols=["col1", "col2"])
        df_encoded = encoder.fit_transform(df)
        assert np.allclose(
            df_encoded["col1_ce"].values, np.array([3, 3, 3, 4, 4, 4, 4,])
        )
        assert df_encoded.columns.tolist() == [
            "col1",
            "col2",
            "target",
            "col1_ce",
            "col2_ce",
        ]


def test_internal_count_encoder():
    X = np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]])
    y = np.array([1, 1, 0, 1, 1, 1, 0])

    encoder = _CountEncoder()
    res = encoder.fit_transform(X[:, 0])
    assert np.array_equal(res, np.array([3, 3, 3, 4, 4, 4, 4,]))
