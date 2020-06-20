"""Module for TargetEncoder."""
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin as SKTransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import column_or_1d, check_is_fitted

from xfeat.types import XDataFrame, XSeries, CSeries, XNDArray
from xfeat.base import TransformerMixin
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
    import cupy  # NOQA
except ImportError:
    cudf = None
    cupy = None


def _get_index(arr, val):
    index = np.searchsorted(arr, val)
    return index


class TargetEncoder(TransformerMixin):
    """Encode categorical values with using target values.

    Example:
        `sklearn.model_selection.KFold` is used to split the training data by default.
        This can be changed by setting `fold` argument.

        ::

            >>> import pandas as pd
            >>> from sklearn.model_selection import KFold
            >>> from xfeat import TargetEncoder
            >>> df = pd.DataFrame({
               "col": ["A", "A", "A", "B", "B", "B", "B"],
               "target": [1, 1, 0, 1, 1, 1, 0],
            })
            >>> fold = KFold(n_splits=2, shuffle=False)
            >>> encoder = TargetEncoder(fold=fold)
            >>> encoder.fit_transform(df)
              col  target    col_te
            0   A       1  0.000000
            1   A       1  0.000000
            2   A       0  0.000000
            3   B       1  0.666667
            4   B       1  1.000000
            5   B       1  1.000000
            6   B       0  1.000000

            >>> df_tst = pd.DataFrame({"col": ["A", "B", "C"]})
            >>> encoder.transform(df_tst)
              col    col_te
            0   A  0.333333
            1   B  0.833333
            2   C  0.000000

    Args:
        input_cols (Optional[List[str]]):
            Input column names. The default uses all columns of the input data frame.

        target_col (str):
            A target column. Defaults to 'target'.

        fold (Optional[KFold]):
            An object to split a dataset. Defaults to None.
            By default `KFold(n_splits=5, shuffle=True, random_state=1)` is used.

        output_prefix (str):
            Prefix of output column name. Defaults: `""`.

        output_suffix (str):
            Suffix of output column name. Defaults: `"_te"`.
    """

    def __init__(
        self,
        input_cols: Optional[List[str]] = None,
        target_col: str = "target",
        fold: Optional[KFold] = None,
        output_prefix: str = "",
        output_suffix: str = "_te",
    ):
        self._input_cols = input_cols or []
        self._target_col = target_col
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._target_encoders: Dict[str, _TargetEncoder] = {}

        if fold is None:
            # Default kfold split
            self.fold = KFold(n_splits=5, shuffle=True, random_state=1)
        elif isinstance(fold, KFold):
            self.fold = fold
        elif isinstance(fold, np.ndarray):
            # TODO(smly): Support np.ndarray input
            raise RuntimeError
        elif isinstance(fold, list):
            # TODO(smly): Support list input
            raise RuntimeError
        else:
            # TODO(smly): Add error message
            raise RuntimeError

    def fit(self, input_df: XDataFrame) -> None:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        # TODO(smly): warn to use fit_transform instead of fit().
        # transform() is recommended for encoding test set.
        input_cols = self._input_cols
        if not input_cols:
            input_cols = input_df.columns.tolist()
            self._input_cols = input_cols

        # Remove `target_col` from `self._input_cols`.
        if self._target_col in self._input_cols:
            self._input_cols.remove(self._target_col)

        for col in self._input_cols:
            target_encoder = _TargetEncoder(self.fold)
            self._target_encoders[col] = target_encoder
            target_encoder.fit(input_df[col], input_df[self._target_col])

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        out_df = input_df.copy()

        for col in self._input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            if isinstance(input_df[col], pd.Series):
                X = column_or_1d(input_df[col], warn=True)
            elif cudf and isinstance(input_df[col], cudf.Series):
                X = input_df[col]
            else:
                raise TypeError

            out_df[out_col] = self._target_encoders[col].transform(X)

        return out_df

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        out_df = input_df.copy()

        input_cols = self._input_cols
        if not input_cols:
            input_cols = input_df.columns.tolist()
            self._input_cols = input_cols

        # Remove `target_col` from `self._input_cols`.
        if self._target_col in self._input_cols:
            self._input_cols.remove(self._target_col)

        for col in self._input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            target_encoder = _TargetEncoder(self.fold)
            self._target_encoders[col] = target_encoder

            if isinstance(input_df[col], pd.Series):
                X = column_or_1d(input_df[col], warn=True)
                y = column_or_1d(input_df[self._target_col], warn=True)
            elif cudf and isinstance(input_df[col], cudf.Series):
                X = input_df[col]
                y = input_df[self._target_col]
            else:
                raise TypeError

            out_df[out_col] = target_encoder.fit_transform(X, y).copy()

        return out_df


class _TargetEncoder(BaseEstimator, SKTransformerMixin):
    """Encode categorical values with ucing target values."""

    def __init__(self, fold: KFold):
        self.fold: KFold = fold

    def fit(self, X: XSeries, y: XSeries) -> None:
        """[summary].

        Args:
            X : [description].
            y (optional): [description]. Defaults to None.
        """
        # TODO(smly): warn to use fit_transform instead of fit().
        # transform() is recommended for encoding test set.
        if cudf_is_available() and isinstance(X, cudf.Series):
            pass
        elif isinstance(X, np.ndarray):
            X = column_or_1d(X, warn=True)
            y = column_or_1d(y, warn=True)
        else:
            raise RuntimeError

        # y = column_or_1d(y, warn=True)
        self.mean_encoders_ = []

        # Fit and append mean_encoders
        for trn_idx, tst_idx in self.fold.split(X):
            X_trn, _ = X[trn_idx], X[tst_idx]
            y_trn, _ = y[trn_idx], y[tst_idx]
            if cudf_is_available() and isinstance(X, cudf.Series):
                encoder = _CuPy_MeanEncoder()
                encoder.fit(X_trn, y_trn)
                self.mean_encoders_.append(encoder)
            elif isinstance(X, np.ndarray):
                encoder = _MeanEncoder()
                encoder.fit(X_trn, y_trn)
                self.mean_encoders_.append(encoder)
            else:
                raise RuntimeError

    def transform(self, X: XSeries) -> XSeries:
        """[summary].

        Args:
            X : [description].
        Returns:
            Any : [description].
        """
        check_is_fitted(self, "mean_encoders_")

        # Encoding for testing part. Different result from `fit_transform()`
        # result.
        if cudf_is_available() and isinstance(X, cudf.Series):
            n_splits = self.fold.get_n_splits()
            likelihood_values = cupy.zeros((X.shape[0], n_splits))
            for fold_idx, mean_encoder in enumerate(self.mean_encoders_):
                ret = mean_encoder.transform(X)
                likelihood_values[:, fold_idx] = ret
            return np.mean(likelihood_values, axis=1)
        else:
            n_splits = self.fold.get_n_splits()
            likelihood_values = np.zeros((X.shape[0], n_splits))
            for fold_idx, mean_encoder in enumerate(self.mean_encoders_):
                ret = mean_encoder.transform(X)
                likelihood_values[:, fold_idx] = ret
            return np.mean(likelihood_values, axis=1)

    def fit_transform(self, X: XSeries, y: XSeries) -> XNDArray:
        """[summary].

        Args:
            X : [description].
        Returns:
            XNDArray : [description].
        """
        self.fit(X, y)
        check_is_fitted(self, "mean_encoders_")

        # Encoding for training data.
        if cudf_is_available() and isinstance(X, cudf.Series):
            likelihood_values = cupy.zeros(X.shape[0])
            for idx, (trn_idx, tst_idx) in enumerate(self.fold.split(X)):
                X_tst = X[tst_idx]
                likelihood_values[tst_idx] = self.mean_encoders_[
                    idx].transform(X_tst)
            return likelihood_values
        elif isinstance(X, np.ndarray):
            likelihood_values = np.zeros(X.shape[0])
            for idx, (trn_idx, tst_idx) in enumerate(self.fold.split(X)):
                X_tst = X[tst_idx]
                likelihood_values[tst_idx] = self.mean_encoders_[
                    idx].transform(X_tst)
            return likelihood_values
        else:
            raise RuntimeError


class _MeanEncoder(BaseEstimator, SKTransformerMixin):
    """Encode categorical values to mean of target values."""

    def __init__(self):
        self.default_unseen_ = 0.0
        self.unseen = "default"

        self._label_encoding_uniques = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """[summary].

        Args:
            X : [description].
            y : [description].
        """
        X = column_or_1d(X, warn=True)
        y = column_or_1d(y, warn=True)

        # Label encoding if necessary
        if not np.can_cast(X.dtype, np.int64):
            X, uniques = pd.Series(X).factorize()
            self._label_encoding_uniques = uniques

        self.classes_, counts = np.unique(X, return_counts=True)
        self.class_means_ = np.zeros_like(self.classes_, dtype="float64")

        for idx, uniq_value in enumerate(self.classes_):
            mean_value = np.mean(y[X == uniq_value])
            self.class_means_[idx] = mean_value

        self.classes_ = np.append(self.classes_, [np.max(self.classes_) + 1])
        self.class_means_ = np.append(
            self.class_means_, [
                self.default_unseen_])

        self.lut_ = np.hstack(
            [self.classes_.reshape(-1, 1), self.class_means_.reshape(-1, 1)]
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """[summary].

        Args:
            X : [description].
        Returns:
            Any : [description].
        """
        check_is_fitted(self, "class_means_")
        X = column_or_1d(X, warn=True)

        # Label encoding if necessary
        if self._label_encoding_uniques is not None:
            X = self._label_encoding_uniques.get_indexer(pd.Series(X))

        missing_mask = np.isnan(X)
        encode_mask = np.invert(missing_mask)
        unseen_mask = np.bitwise_xor(
            np.isin(X, self.classes_, invert=True), missing_mask
        )

        X = X.copy()
        X[unseen_mask] = np.max(self.classes_)

        indices = _get_index(self.classes_, X[encode_mask])

        _classes_index_list = np.searchsorted(self.lut_[:, 0], self.classes_)
        encoded_values = np.zeros(X.shape[0], dtype=np.float32)
        encoded_values[encode_mask] = np.take(
            self.lut_[:, 1], np.take(_classes_index_list, indices)
        )

        encoded_values[unseen_mask] = self.default_unseen_
        return encoded_values

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[summary].

        Args:
            X : [description].
            y : [description].
        Returns:
            Any : [description].
        """
        X = column_or_1d(X, warn=True)
        y = column_or_1d(y, warn=True)

        self.fit(X, y)
        return self.transform(X)


class _CuPy_MeanEncoder(BaseEstimator):
    """Encode categorical values to mean of target values."""

    def __init__(self):
        self.default_unseen_ = 0.0
        self.unseen = "default"

        self._label_encoding_uniques = None

    def fit(self, X: CSeries, y: CSeries):
        """[summary].

        Args:
            X (cupy.ndarray): Input cupy ndarray.
            y (cupy.ndarray): Target cupy ndarray.
        """
        # Label encoding if necessary
        if not cupy.can_cast(X.dtype, cupy.int):
            X, uniques = pd.Series(cupy.asnumpy(X)).factorize()
            X = cudf.Series(X)
            self._label_encoding_uniques = uniques

        self.classes_, counts = cupy.unique(X, return_counts=True)
        self.class_means_ = cupy.zeros_like(self.classes_, dtype="float64")

        assert isinstance(y, cudf.Series)
        df = cudf.DataFrame()
        df.insert(0, "X", X)
        df.insert(0, "y", y.values)
        agg = df.groupby("X").agg("mean").to_pandas()

        for idx, uniq_value in enumerate(self.classes_):
            uniq_value = cupy.asnumpy(uniq_value).item()
            mean_value = agg.loc[uniq_value]["y"]
            self.class_means_[idx] = mean_value

        self.classes_ = cupy.array(
            np.append(cupy.asnumpy(self.classes_),
                      [cupy.asnumpy(cupy.max(self.classes_)) + 1])
        )
        self.class_means_ = cupy.array(
            np.append(cupy.asnumpy(self.class_means_),
                      [cupy.asnumpy(self.default_unseen_)])
        )

        self.lut_ = cupy.hstack(
            [self.classes_.reshape(-1, 1), self.class_means_.reshape(-1, 1)]
        )

    def transform(self, X):
        """[summary].

        Args:
            X (cupy.ndarray): [description].
        Returns:
            cupy.ndarray: [description].
        """
        check_is_fitted(self, "class_means_")
        # TODO(smly):
        # X = column_or_1d(X, warn=True)

        # Label encoding if necessary
        if self._label_encoding_uniques is not None:
            X = self._label_encoding_uniques.get_indexer(X.to_pandas())
        X = cupy.asarray(X)

        missing_mask = cupy.isnan(X)
        encode_mask = cupy.invert(missing_mask)
        unseen_mask = cupy.bitwise_xor(
            cupy.isin(X, self.classes_, invert=True), missing_mask
        )

        X = X.copy()
        X[unseen_mask] = cupy.max(self.classes_)

        indices = _get_index(self.classes_, X[encode_mask])

        _classes_index_list = cupy.searchsorted(self.lut_[:, 0], self.classes_)
        encoded_values = cupy.zeros(X.shape[0], dtype=cupy.float32)
        encoded_values[encode_mask] = cupy.take(
            self.lut_[:, 1], cupy.take(_classes_index_list, indices)
        )

        encoded_values[unseen_mask] = self.default_unseen_
        return encoded_values

    def fit_transform(self, X, y):
        """[summary].

        Args:
            X : [description].
            y : [description].
        Returns:
            Any : [description].
        """
        # TODO(smly):
        # X = column_or_1d(X, warn=True)
        # y = column_or_1d(y, warn=True)
        self.fit(X, y)
        return self.transform(X)
