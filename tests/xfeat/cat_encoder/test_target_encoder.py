import pytest

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from xfeat import TargetEncoder
from xfeat.cat_encoder._target_encoder import _MeanEncoder
from xfeat.cat_encoder._target_encoder import _CuPy_MeanEncoder
from xfeat.cat_encoder._target_encoder import _TargetEncoder
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
    import cupy  # NOQA
except ImportError:
    cudf = None
    cupy = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame(
        {
            "col1": ["2", "2", "2", "8", "8", "8", "8"],
            "col2": [2, 4, 6, 7, 8, 9, 10],
            "target": [1, 1, 0, 1, 1, 1, 0],
        }
    )

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_target_encoder_with_categorical_values(dataframes):
    for df in dataframes:
        fold = KFold(n_splits=2, shuffle=False)
        encoder = TargetEncoder(input_cols=["col1", "col2"], fold=fold)
        df_encoded = encoder.fit_transform(df)

        assert encoder.fold.get_n_splits() == 2
        assert list(sorted(encoder._target_encoders.keys())) == ["col1", "col2"]

        assert np.allclose(
            df_encoded["col1_te"].values,
            np.array([0.0, 0.0, 0.0, 0.66666667, 1.0, 1.0, 1.0,]),
        )
        assert df.columns.tolist() == [
            "col1",
            "col2",
            "target",
        ]
        assert df_encoded.columns.tolist() == [
            "col1",
            "col2",
            "target",
            "col1_te",
            "col2_te",
        ]


@pytest.fixture
def dataframes_targetencoder():
    df = pd.DataFrame(
        {
            "col1": [2, 2, 2, 8, 8, 8, 8],
            "col2": [2, 4, 6, 7, 8, 9, 10],
            "target": [1, 1, 0, 1, 1, 1, 0],
        }
    )
    df_test = pd.DataFrame({
        "col1": [2, 8],
        "col2": [2, 8],
    })

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        df_test_cuda = cudf.from_pandas(df_test)
        return [(df, df_test), (df_cuda, df_test_cuda)]
    else:
        return [(df, df_test)]


def test_target_encoder(dataframes_targetencoder):
    for df, df_test in dataframes_targetencoder:
        fold = KFold(n_splits=2, shuffle=False)
        encoder = TargetEncoder(input_cols=["col1", "col2"], fold=fold)
        df_encoded = encoder.fit_transform(df)
        assert np.allclose(
            df_encoded["col1_te"].values,
            np.array([0.0, 0.0, 0.0, 0.66666667, 1.0, 1.0, 1.0,])
        )
        assert df_encoded.columns.tolist() == [
            "col1",
            "col2",
            "target",
            "col1_te",
            "col2_te",
        ]
        assert df.columns.tolist() == [
            "col1",
            "col2",
            "target",
        ]

        df_test_encoded = encoder.transform(df_test)

        assert np.allclose(
            df_test_encoded["col1_te"].values,
            np.array([0.333333, 0.833333])
        )
        assert np.allclose(
            df_test_encoded["col2_te"].values,
            np.array([0.5, 0.5])
        )
        assert df_test_encoded.columns.tolist() == [
            "col1",
            "col2",
            "col1_te",
            "col2_te",
        ]
        assert df_test.columns.tolist() == [
            "col1",
            "col2",
        ]


def test_internal_target_encoder():
    X = np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]])
    y = np.array([1, 1, 0, 1, 1, 1, 0])

    fold = KFold(n_splits=2, shuffle=False)
    trn_idx, tst_idx = next(fold.split(X))
    assert np.array_equal(tst_idx, np.array([0, 1, 2, 3]))

    encoder = _TargetEncoder(fold=fold)

    # Test `fit_transform()`.
    y_trn = encoder.fit_transform(X[:, 0], y)
    assert np.allclose(y_trn, np.array([0.0, 0.0, 0.0, 0.66666667, 1.0, 1.0, 1.0,]))

    X_tst = np.array([8, 0, 2])
    y_tst = encoder.transform(X_tst)
    assert np.allclose(
        y_tst,
        np.array(
            [
                0.83333334, 0.        , 0.33333334
            ]
        ),
    )


def test_internal_target_encoder_with_cudf():
    if not cudf_is_available() or cudf is not None or cupy is not None:
        # Skip test.
        return

    X = cudf.Series(np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]]))
    y = cudf.Series(np.array([1, 1, 0, 1, 1, 1, 0]))

    fold = KFold(n_splits=2, shuffle=False)
    trn_idx, tst_idx = next(fold.split(X))
    assert np.array_equal(tst_idx, np.array([0, 1, 2, 3]))

    encoder = _TargetEncoder(fold=fold)

    # Test `fit_transform()`.
    y_trn = encoder.fit_transform(X[:, 0], y)
    assert np.allclose(y_trn.values, np.array([0.0, 0.0, 0.0, 0.66666667, 1.0, 1.0, 1.0,]))

    X_tst = np.array([8, 0, 2])
    y_tst = encoder.transform(X_tst)
    assert np.allclose(
        y_tst.values,
        np.array(
            [
                0.83333334, 0.        , 0.33333334
            ]
        ),
    )

def test_internal_target_encoder_fit_and_transform():
    X = np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]])
    y = np.array([1, 1, 0, 1, 1, 1, 0])

    fold = KFold(n_splits=2, shuffle=False)
    trn_idx, tst_idx = next(fold.split(X))
    assert np.array_equal(tst_idx, np.array([0, 1, 2, 3]))

    encoder = _TargetEncoder(fold=fold)

    # Test `fit()` and `fit_transform()`.
    encoder.fit(X[:, 0], y)
    y_trn = encoder.transform(X[:, 0])
    assert np.allclose(y_trn, np.array([
        0.33333334, 0.33333334, 0.33333334, 0.83333334, 0.83333334, 0.83333334, 0.83333334,
    ]))

    X_tst = np.array([8, 0, 2])
    y_tst = encoder.transform(X_tst)
    assert np.allclose(
        y_tst,
        np.array(
            [
                0.83333334, 0.        , 0.33333334
            ]
        ),
    )


def test_internal_target_encoder_fit_and_transform_with_cudf():
    if not cudf_is_available() or cudf is not None or cupy is not None:
        # Skip test.
        return

    X = cudf.Series(np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]]))
    y = cudf.Series(np.array([1, 1, 0, 1, 1, 1, 0]))

    fold = KFold(n_splits=2, shuffle=False)
    trn_idx, tst_idx = next(fold.split(X))
    assert np.array_equal(tst_idx, np.array([0, 1, 2, 3]))

    encoder = _TargetEncoder(fold=fold)

    # Test `fit()` and `fit_transform()`.
    encoder.fit(X[:, 0], y)
    y_trn = encoder.transform(X[:, 0])
    assert np.allclose(y_trn.values, np.array([
        0.33333334, 0.33333334, 0.33333334, 0.83333334, 0.83333334, 0.83333334, 0.83333334,
    ]))

    X_tst = np.array([8, 0, 2])
    y_tst = encoder.transform(X_tst)
    assert np.allclose(
        y_tst.values,
        np.array(
            [
                0.83333334, 0.        , 0.33333334
            ]
        ),
    )


def test_internal_mean_encoder():
    X = np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]])
    y = np.array([1, 1, 0, 0, 1, 1, 0])

    col_idx = 0
    encoder = _MeanEncoder()
    encoder.fit(X[:, col_idx], y)
    y_mean = encoder.transform(X[:, col_idx])

    assert np.array_equal(
        encoder.classes_,
        np.array([2, 8, 9]),  # 9 (max + 1) is assigned for unseen values.
    )
    assert np.allclose(
        encoder.class_means_, np.array([0.66666667, 0.5, 0.0,])  # 2/3  # 2/4
    )
    assert np.allclose(
        y_mean, np.array([0.66666667, 0.66666667, 0.66666667, 0.5, 0.5, 0.5, 0.5,])
    )

    # Unseen values
    col_idx = 0
    X_test = np.array([9, 1, 8, 2])
    y_mean = encoder.transform(X_test)

    assert np.allclose(
        y_mean,
        np.array(
            [
                0.0,  # 9 = recognized as seen value since (max+1) is assigned for unseen value.
                0.0,  # 1 = unseen value
                0.5,  # 8 = 2/4
                0.66666667,
            ]
        ),
    )

    # Missing value
    col_idx = 0
    X_test = np.array([[np.nan, 2], [1, 1], [8, 4]])
    y_mean = encoder.transform(X_test[:, col_idx])

    assert np.allclose(
        y_mean,
        np.array(
            [0.0, 0.0, 0.5,]  # NaN = missing value  # 1 = unseen value  # 0 = 2/4
        ),
    )


def test_internal_mean_encoder_fit_transform():
    X = np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]])
    y = np.array([1, 1, 0, 0, 1, 1, 0])

    col_idx = 0
    encoder = _MeanEncoder()
    y_mean = encoder.fit_transform(X[:, col_idx], y)

    assert np.array_equal(
        encoder.classes_,
        np.array([2, 8, 9]),  # 9 (max + 1) is assigned for unseen values.
    )
    assert np.allclose(
        encoder.class_means_, np.array([0.66666667, 0.5, 0.0,])  # 2/3  # 2/4
    )
    assert np.allclose(
        y_mean, np.array([0.66666667, 0.66666667, 0.66666667, 0.5, 0.5, 0.5, 0.5,])
    )

    # Unseen values
    col_idx = 0
    X_test = np.array([9, 1, 8, 2])
    y_mean = encoder.transform(X_test)

    assert np.allclose(
        y_mean,
        np.array(
            [
                0.0,  # 9 = recognized as seen value since (max+1) is assigned for unseen value.
                0.0,  # 1 = unseen value
                0.5,  # 8 = 2/4
                0.66666667,
            ]
        ),
    )

    # Missing value
    col_idx = 0
    X_test = np.array([[np.nan, 2], [1, 1], [8, 4]])
    y_mean = encoder.transform(X_test[:, col_idx])

    assert np.allclose(
        y_mean,
        np.array(
            [0.0, 0.0, 0.5,]  # NaN = missing value  # 1 = unseen value  # 0 = 2/4
        ),
    )


def test_internal_cupy_mean_encoder():
    if not cudf_is_available() or cudf is not None or cupy is not None:
        # Skip test.
        return

    X = np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]])
    y = np.array([1, 1, 0, 0, 1, 1, 0])

    X = cupy.asarray(X)
    y = cupy.asarray(y)

    col_idx = 0
    encoder = _CuPy_MeanEncoder()
    encoder.fit(X[:, col_idx], y)
    y_mean = encoder.transform(X[:, col_idx])

    assert np.array_equal(
        cupy.asnumpy(encoder.classes_),
        np.array([2, 8, 9]),  # 9 (max + 1) is assigned for unseen values.
    )
    assert cupy.allclose(
        encoder.class_means_, cupy.array([0.66666667, 0.5, 0.0,])  # 2/3  # 2/4
    )
    assert cupy.allclose(
        y_mean, cupy.array([0.66666667, 0.66666667, 0.66666667, 0.5, 0.5, 0.5, 0.5,])
    )

    # Unseen values
    col_idx = 0
    X_test = cupy.array([9, 1, 8, 2])
    y_mean = encoder.transform(X_test)

    assert cupy.allclose(
        y_mean,
        cupy.array(
            [
                0.0,  # 9 = recognized as seen value since (max+1) is assigned for unseen value.
                0.0,  # 1 = unseen value
                0.5,  # 8 = 2/4
                0.66666667,
            ]
        ),
    )

    # Missing value
    col_idx = 0
    X_test = cupy.array([[cupy.nan, 2], [1, 1], [8, 4]])
    y_mean = encoder.transform(X_test[:, col_idx])

    assert cupy.allclose(
        y_mean,
        cupy.array(
            [0.0, 0.0, 0.5,]  # NaN = missing value  # 1 = unseen value  # 0 = 2/4
        ),
    )


def test_internal_cupy_mean_encoder_fit_transform():
    if not cudf_is_available() or cudf is not None or cupy is not None:
        # Skip test.
        return

    X = np.array([[2, 2], [2, 4], [2, 6], [8, 7], [8, 8], [8, 9], [8, 10]])
    y = np.array([1, 1, 0, 0, 1, 1, 0])

    X = cupy.asarray(X)
    y = cupy.asarray(y)

    col_idx = 0
    encoder = _CuPy_MeanEncoder()
    y_mean = encoder.fit_transform(X[:, col_idx], y)

    assert np.array_equal(
        cupy.asnumpy(encoder.classes_),
        np.array([2, 8, 9]),  # 9 (max + 1) is assigned for unseen values.
    )
    assert cupy.allclose(
        encoder.class_means_, cupy.array([0.66666667, 0.5, 0.0,])  # 2/3  # 2/4
    )
    assert cupy.allclose(
        y_mean, cupy.array([0.66666667, 0.66666667, 0.66666667, 0.5, 0.5, 0.5, 0.5,])
    )

    # Unseen values
    col_idx = 0
    X_test = cupy.array([9, 1, 8, 2])
    y_mean = encoder.transform(X_test)

    assert cupy.allclose(
        y_mean,
        cupy.array(
            [
                0.0,  # 9 = recognized as seen value since (max+1) is assigned for unseen value.
                0.0,  # 1 = unseen value
                0.5,  # 8 = 2/4
                0.66666667,
            ]
        ),
    )

    # Missing value
    col_idx = 0
    X_test = cupy.array([[cupy.nan, 2], [1, 1], [8, 4]])
    y_mean = encoder.transform(X_test[:, col_idx])

    assert cupy.allclose(
        y_mean,
        cupy.array(
            [0.0, 0.0, 0.5,]  # NaN = missing value  # 1 = unseen value  # 0 = 2/4
        ),
    )
