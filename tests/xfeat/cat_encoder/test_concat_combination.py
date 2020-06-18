import pytest

import pandas as pd

from xfeat.cat_encoder import ConcatCombination
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"col1": ["a", "b"], "col2": ["@", "%"], "col3": ["X", "Y"],})

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_concat_combination(dataframes):
    for df in dataframes:
        encoder = ConcatCombination()
        df_encoded = encoder.fit_transform(df)
        assert df_encoded.columns.tolist() == [
            "col1",
            "col2",
            "col3",
            "col1col2_combi",
            "col1col3_combi",
            "col2col3_combi",
        ]
        assert df_encoded["col1col3_combi"].tolist() == ["aX", "bY"]

    for df in dataframes:
        encoder = ConcatCombination(output_suffix="", drop_origin=True)
        df_encoded = encoder.fit_transform(df)
        assert df_encoded.columns.tolist() == [
            "col1col2",
            "col1col3",
            "col2col3",
        ]
        assert df_encoded["col2col3"].tolist() == ["@X", "%Y"]

    for df in dataframes:
        encoder = ConcatCombination(output_suffix="", drop_origin=True, r=3)
        df_encoded = encoder.fit_transform(df)
        assert df_encoded.columns.tolist() == [
            "col1col2col3",
        ]
        assert df_encoded["col1col2col3"].tolist() == ["a@X", "b%Y"]
