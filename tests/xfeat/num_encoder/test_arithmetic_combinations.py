import pytest
import pandas as pd

from xfeat.utils import cudf_is_available
from xfeat.num_encoder import ArithmeticCombinations


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame(
        {"col1": [1, 2, 3, 4, 5], "col2": [2, 3, 4, 5, 6], "col3": [3, 4, 5, 6, 7],}
    )

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]
    return [df]


def test_arithmetic_combinations(dataframes):
    for df in dataframes:
        encoder = ArithmeticCombinations(operator="+", output_suffix="_plus")
        df_new = encoder.fit_transform(df)

        assert df_new.columns.tolist() == [
            "col1",
            "col2",
            "col3",
            "col1col2_plus",
            "col1col3_plus",
            "col2col3_plus",
        ]
        assert df_new["col2col3_plus"].tolist() == [5, 7, 9, 11, 13]
