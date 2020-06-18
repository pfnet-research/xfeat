import pytest
import numpy as np
import pandas as pd

from xfeat.utils import cudf_is_available
from xfeat import UserDefinedLabelEncoder


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({
        "color": ["red", "red", "yellow"],
    })

    # TODO(smly): Need to fix here for cuDF.
    # if cudf_is_available():
    #     df_cuda = cudf.from_pandas(df)
    #     return [df, df_cuda]
    # else:
    return [df]


def test_user_defined_label_encoder(dataframes):
    label_mapping = {
        "red": 1001,
        "blue": 1002,
        "yellow": 1003,
    }

    for df in dataframes:
        encoder = UserDefinedLabelEncoder(label_mapping)
        df_encoded = encoder.fit_transform(df)
        assert df_encoded["color_le"].tolist() == [1001, 1001, 1003]


def test_user_defined_label_encoder_with_unseen(dataframes):
    label_mapping = {
        "red": 1001,
        "blue": 1002,
        "yellow": 1003,
    }

    df = pd.DataFrame({
        "color": ["red", "red", "yellow"],
    })

    encoder = UserDefinedLabelEncoder(label_mapping)
    df_encoded = encoder.fit_transform(df)
    assert df_encoded["color_le"].tolist() == [1001, 1001, 1003]

    df_test = pd.DataFrame({
        "color": ["red", "red", "yellow", "__unseen__"],
    })
    df_test_encoded = encoder.transform(df_test)
    assert df_test_encoded["color_le"].tolist() == [1001, 1001, 1003, -1]
