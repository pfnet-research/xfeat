import pytest

import numpy as np
import pandas as pd

from xfeat.generic_encoder import LambdaEncoder
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"col1": [1, 2, 3],})

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_lambda_encoder(dataframes):
    for df in dataframes:
        encoder = LambdaEncoder(lambda x: x + 1, fillna=0)
        df_encoded = encoder.fit_transform(df)
        print(
            df_encoded["col1_lmd"].values,
            df_encoded["col1_lmd"].dtype,
            type(df_encoded["col1_lmd"]),
        )
        print(np.array([2, 3, 4]))
        assert df_encoded.columns.tolist() == ["col1", "col1_lmd"]
        assert np.allclose(df_encoded["col1_lmd"].values, np.array([2, 3, 4]))
