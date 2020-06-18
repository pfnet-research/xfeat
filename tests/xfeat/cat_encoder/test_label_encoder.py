import pytest
import numpy as np
import pandas as pd

from xfeat.utils import cudf_is_available
from xfeat.cat_encoder import LabelEncoder


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"col": ["a", "a", "b"],})

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_label_encoder_cat_cols(dataframes):
    for df in dataframes:
        encoder = LabelEncoder()
        df_encoded = encoder.fit_transform(df)
        assert np.allclose(df_encoded["col_le"].values, np.array([0, 0, 1]))
        assert df_encoded.columns.tolist() == [
            "col",
            "col_le",
        ]


def test_label_encoder_sort_category_before_factorize():
    df = pd.DataFrame({"col": ["b", "b", "a"],})

    encoder = LabelEncoder(sort_category=True)
    df_encoded = encoder.fit_transform(df)
    assert np.allclose(df_encoded["col_le"], np.array([1, 1, 0]))
    assert df_encoded.columns.tolist() == [
        "col",
        "col_le",
    ]


def test_label_encoder_with_missing_values():
    df = pd.DataFrame({"col": ["b", "a", "c", np.nan, np.nan],})

    encoder = LabelEncoder(sort_category=True)
    df_encoded = encoder.fit_transform(df)
    assert np.allclose(df_encoded["col_le"], np.array([1, 0, 2, -1, -1]))
    assert df_encoded.columns.tolist() == [
        "col",
        "col_le",
    ]


def test_label_encoder_with_unseen_values():
    df_trn = pd.DataFrame({"col": ["b", "a", "c", np.nan, np.nan],})
    df_tst = pd.DataFrame({"col": ["x"],})

    encoder_1 = LabelEncoder(sort_category=True, output_suffix="", unseen="minus_one")
    df_trn_1 = encoder_1.fit_transform(df_trn)
    assert np.allclose(df_trn_1["col"], np.array([1, 0, 2, -1, -1]))
    assert df_trn_1.columns.tolist() == ["col"]

    df_tst_1 = encoder_1.transform(df_tst)
    assert np.allclose(df_tst_1["col"], np.array([-1]))
    assert df_tst_1.columns.tolist() == ["col"]

    encoder_2 = LabelEncoder(sort_category=True, output_suffix="", unseen="n_unique")
    df_trn_2 = encoder_2.fit_transform(df_trn)
    assert np.allclose(df_trn_2["col"], np.array([1, 0, 2, -1, -1]))
    assert df_trn_2.columns.tolist() == ["col"]

    df_tst_2 = encoder_2.transform(df_tst)
    assert np.allclose(df_tst_2["col"], np.array([3]))
    assert df_tst_2.columns.tolist() == ["col"]
