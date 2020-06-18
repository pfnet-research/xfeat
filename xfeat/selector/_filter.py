"""Filter-based feature selection methods."""
from typing import List
from logging import getLogger

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

from xfeat.utils import analyze_columns, is_cudf
from xfeat.base import SelectorMixin
from xfeat.types import XDataFrame


logger = getLogger(__name__)


class BaseSelectorKBest(SelectorMixin):
    """Base class for k-best feature selector.

    Args:
        input_cols (optional): [description]. Defaults to None.
        target_col (optional): [description]. Defaults to 'target'.
        k (optional): [description]. Defaults to None.
    """

    def __init__(self, input_cols=None, target_col="target", k=None):
        """[summary]."""
        self._input_cols = input_cols
        self._target_col = target_col
        self._k = k

        self._internal_selector = None

    def _get_function(self):
        raise NotImplementedError

    def _lazy_setup_input_cols(self, input_cols):
        if self._input_cols is None:
            self._input_cols = [col for col in input_cols if col != self._target_col]

    def get_selected_cols(self) -> List[str]:
        """Get selected column names.

        Returns:
            selected_cols (List[str]) : Selected column names.
        """
        return [
            col
            for col, is_selected in zip(
                self._input_cols, self._internal_selector.get_support()
            )
            if is_selected
        ]

    def reset_k(self, k: int):
        """Reset parameter k to internal selector.

        Args:
            k (int): Parameter k.
        """
        self._k = k
        if self._internal_selector:
            self._internal_selector.k = k

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        self._lazy_setup_input_cols(input_df.columns.tolist())

        if self._k is None:
            raise RuntimeError("Keyword argument `k` is required.")

        self._internal_selector = SelectKBest(self._get_function(), k=self._k)

        if is_cudf(input_df):
            # TODO(smly): Make this faster using GPUs
            self._internal_selector.fit(
                input_df[self._input_cols].to_pandas(),
                input_df[self._target_col].to_pandas(),
            )
            return self.transform(input_df)
        else:
            self._internal_selector.fit(
                input_df[self._input_cols], input_df[self._target_col]
            )
            return self.transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        if self._internal_selector is None:
            raise RuntimeError("Call fit_transform() before transform().")

        return input_df[self.get_selected_cols()]


class ChiSquareKBest(BaseSelectorKBest):
    """[summary]."""

    def _get_function(self):
        return chi2


class ANOVAClassifKBest(BaseSelectorKBest):
    """[summary]."""

    def _get_function(self):
        return f_classif


class ANOVARegressionKBest(BaseSelectorKBest):
    """[summary]."""

    def _get_function(self):
        return f_regression


class MutualInfoClassifKBest(BaseSelectorKBest):
    """[summary]."""

    def _get_function(self):
        return mutual_info_classif


class SpearmanCorrelationEliminator(SelectorMixin):
    # TODO(smly): SelectorMixin -> BaseSelectorKBest に変更して ~~KBest class にする
    """[summary].

    Args:
        threshold (optional): [description]. Defaults to 0.99.
    """

    def __init__(self, threshold=0.99):
        """[summary]."""
        self._selected_cols = []
        self._threshold = threshold

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        num_cols, cat_cols = analyze_columns(input_df)

        if len(cat_cols) > 0:
            logger.warning("All categorical columns are not eliminated.")

        removed_cols = []
        df_corr = input_df[num_cols].corr()
        corr_cols = df_corr.columns.tolist()
        for i in range(df_corr.shape[0]):
            for j in range(i):

                if abs(df_corr.iloc[i, j]) > self._threshold:
                    removed_cols.append(corr_cols[i])

        self._selected_cols = [col for col in num_cols if col not in set(removed_cols)]
        self._selected_cols += cat_cols
        return self.transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[self._selected_cols]

    def get_selected_cols(self) -> List[str]:
        """Get selected column names.

        Returns:
            selected_cols (List[str]) : Selected column names.
        """
        return self._selected_cols
