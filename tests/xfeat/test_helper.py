import pytest

import pandas as pd
import numpy as np

from xfeat.utils import cudf_is_available
from xfeat.helper import aggregation


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": ["a", "a", "a", "b", "b"], "c": [0, 0, 1, 1, 1],}
    )

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_aggregation(dataframes):
    for input_df in dataframes:
        group_key = "b"
        group_values = ["a", "c"]
        agg_methods = ["max"]

        new_df, new_cols = aggregation(input_df, group_key, group_values, agg_methods)
        assert new_cols == ["agg_max_a_grpby_b", "agg_max_c_grpby_b"]
        assert "agg_max_a_grpby_b" in new_df.columns
        assert "agg_max_c_grpby_b" in new_df.columns
        assert np.allclose(
            new_df["agg_max_a_grpby_b"].values, np.array([3, 3, 3, 5, 5])
        )
        assert np.allclose(
            new_df["agg_max_c_grpby_b"].values, np.array([1, 1, 1, 1, 1])
        )
