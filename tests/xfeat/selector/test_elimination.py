import pytest

import pandas as pd
import numpy as np

from xfeat.selector import SpearmanCorrelationEliminator
from xfeat.selector import DuplicatedFeatureEliminator
from xfeat.selector import ConstantFeatureEliminator
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "d": [1, 2, 9],})

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_duplicated_feature_eliminator(dataframes):
    for df in dataframes:
        selector = DuplicatedFeatureEliminator()
        selector.fit_transform(df)

        assert selector._selected_cols == ["a", "d"]


def test_constant_feature_eliminator():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1],
            "b": [1, 1, np.nan],
            "c": [9, 9, 9],
            "d": [1, 2, 3],
            "e": ["a", "b", "c"],
            "f": ["a", "a", "a"],
            "g": ["a", np.nan, np.nan],
            "h": [np.nan, np.nan, np.nan],
        }
    )

    selector = ConstantFeatureEliminator()
    df_selected = selector.fit_transform(df)

    assert selector._selected_cols == ["b", "d", "e", "g"]
    assert df_selected.columns.tolist() == ["b", "d", "e", "g"]


def test_spearman_correlation_feature_eliminator():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 1, 1, 9],
            "b": [1, 4, 6, 9, 1, 1, 1, np.nan],
            "c": [9, 9, 9, 9, 9, 9, 9, np.nan],
            "d": [1, 2, 3, 4, 1, 1, 1, 1],
        }
    )

    selector = SpearmanCorrelationEliminator(threshold=0.99)
    selector.fit_transform(df)

    assert len(selector._selected_cols) == 3
    assert selector._selected_cols == ["a", "b", "c"]
