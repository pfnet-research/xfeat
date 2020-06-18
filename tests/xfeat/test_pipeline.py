import pytest

import numpy as np
import pandas as pd

from xfeat.base import TransformerMixin
from xfeat.pipeline import Pipeline
from xfeat.utils import cudf_is_available
from xfeat.types import XDataFrame


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


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
        assert np.allclose(df["new1"].values, np.array([1, 1, 1]))
        assert np.allclose(df["new2"].values, np.array([2, 2, 2]))
