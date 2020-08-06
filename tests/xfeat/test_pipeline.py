import pytest

import numpy as np
import pandas as pd

from xfeat.types import XSeries
from xfeat.base import TransformerMixin
from xfeat.pipeline import Pipeline
from xfeat.utils import cudf_is_available
from xfeat.types import XDataFrame


try:
    import cudf  # NOQA
    import cupy as cp  # NOQA
except ImportError:
    cudf = None
    cp = None


def _allclose(lhs: XSeries, rhs: np.ndarray):
    if cudf_is_available():
        return np.allclose(cp.asnumpy(lhs.values), rhs)
    else:
        return np.allclose(lhs.values, rhs)


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"var1": [1, 2, 3]})

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_pipeline(dataframes):
    class DummyTransformer1(TransformerMixin):
        def transform(self, input_df: XDataFrame) -> XDataFrame:
            input_df["new1"] = 1
            return input_df

    class DummyTransformer2(TransformerMixin):
        def transform(self, input_df: XDataFrame) -> XDataFrame:
            input_df["new2"] = 2
            return input_df

    for df in dataframes:
        pipeline = Pipeline([DummyTransformer1(), DummyTransformer2()])
        df = pipeline.transform(df)
        assert df.columns.tolist() == ["var1", "new1", "new2"]
        assert _allclose(df["new1"], np.array([1, 1, 1]))
        assert _allclose(df["new2"], np.array([2, 2, 2]))
