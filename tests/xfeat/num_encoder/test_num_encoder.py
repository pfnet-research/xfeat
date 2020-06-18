import pytest
import pandas as pd

from xfeat import SelectNumerical
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def pandas_dataframe():
    df = pd.DataFrame({"col": ["A", "B", "B"], "num": [1, 2, 3],})

    return df


def test_select_numerical(pandas_dataframe):
    encoder = SelectNumerical()
    df_new = encoder.fit_transform(pandas_dataframe)
    assert df_new.columns.tolist() == ["num"]


def test_select_numerical_cudf(pandas_dataframe):
    if not cudf_is_available():
        return

    df_cuda = cudf.from_pandas(pandas_dataframe)
    encoder = SelectNumerical()
    df_new = encoder.fit_transform(df_cuda)
    assert df_new.columns.tolist() == ["num"]
