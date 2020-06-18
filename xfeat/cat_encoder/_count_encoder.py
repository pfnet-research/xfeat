"""Modue for CountEncoder."""
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin as SKTransformerMixin
from sklearn.utils.validation import column_or_1d, check_is_fitted

from xfeat.types import XDataFrame
from xfeat.base import TransformerMixin


class CountEncoder(TransformerMixin):
    """Encode frequency of categorical values.

    Example:
        ::

            >>> import pandas as pd
            >>> from xfeat import CountEncoder
            >>> df = pd.DataFrame({"col": [9, 1, 1]})
            >>> encoder = CountEncoder()
            >>> encoder.fit_transform(df)
               col  col_ce
            0    9       1
            1    1       2
            2    1       2

            >>> df = pd.DataFrame({"col": ["A", "A", "B"]})
            >>> encoder = CountEncoder(output_suffix="")
            >>> encoder.fit_transform(df)
               col
            0    2
            1    2
            2    1

    Args:
        input_cols (Optional[List[str]]):
            Input column names. The default uses all columns of the input data frame.

        output_prefix (str):
            Prefix of output column name. Defaults: `""`.

        output_suffix (str):
            Suffix of output column name. Defaults: `"_ce"`.
    """

    def __init__(
        self,
        input_cols: Optional[List[str]] = None,
        output_prefix: str = "",
        output_suffix: str = "_ce",
    ):
        self._input_cols = input_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._count_encoders: Dict[str, _CountEncoder] = {}

    def fit(self, input_df: XDataFrame) -> None:
        """Transform data frame.

        Args:
            input_df (XDataFrame):
                Input data frame.
        """
        input_cols = self._input_cols

        if not input_cols:
            input_cols = input_df.columns.tolist()
            self._input_cols = input_cols

        for col in self._input_cols:
            count_encoder = _CountEncoder()
            self._count_encoders[col] = count_encoder

            count_encoder.fit(input_df[col])

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.

        Returns:
            XDataFrame: Output data frame.
        """
        new_df = input_df.copy()

        for col in self._input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            count_encoder = self._count_encoders[col]
            new_df[out_col] = count_encoder.transform(new_df[col].copy())

        return new_df

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.

        Returns:
            XDataFrame: Output data frame.
        """
        new_df = input_df.copy()

        input_cols = self._input_cols
        if not input_cols:
            input_cols = new_df.columns.tolist()
            self._input_cols = input_cols

        for col in self._input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            count_encoder = _CountEncoder()
            self._count_encoders[col] = count_encoder
            new_df[out_col] = count_encoder.fit_transform(new_df[col].copy())

        return new_df


class _CountEncoder(BaseEstimator, SKTransformerMixin):
    def __init__(self):
        self._default_unseen = 1
        self._default_missing = 1

        self._label_encoding_uniques = None

    def fit(self, X, y=None):
        """Fit to ndarray, then transform it."""
        X = column_or_1d(X, warn=True)

        # Label encoding if necessary
        if not np.can_cast(X.dtype, np.int64):
            X, uniques = pd.Series(X).factorize()
            self._label_encoding_uniques = uniques

        self.classes_, self.counts_ = np.unique(X[np.isfinite(X)], return_counts=True)

        self.classes_ = np.append(self.classes_, [np.max(self.classes_) + 1])
        self.counts_ = np.append(self.counts_, [self._default_unseen])
        self.lut_ = np.hstack(
            [self.classes_.reshape(-1, 1), self.counts_.reshape(-1, 1)]
        )
        return self

    def transform(self, X):
        """Transform ndarray values."""
        check_is_fitted(self, "classes_")
        X = column_or_1d(X, warn=True)

        # Label encoding if necessary
        if self._label_encoding_uniques is not None:
            X = self._label_encoding_uniques.get_indexer(pd.Series(X))

        missing_mask = np.isnan(X)
        encode_mask = np.invert(missing_mask)
        unseen_mask = np.bitwise_xor(
            np.isin(X, self.classes_, invert=True), missing_mask
        )

        X[unseen_mask] = np.max(self.classes_)
        indices = np.searchsorted(self.classes_, X[encode_mask])

        X[encode_mask] = np.take(
            self.lut_[:, 1],
            np.take(np.searchsorted(self.lut_[:, 0], self.classes_), indices),
        )

        if np.any(missing_mask):
            X[missing_mask] = self._default_missing

        return X
