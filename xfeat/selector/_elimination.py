"""Feature elimination methods."""
from xfeat.base import SelectorMixin
from xfeat.types import XDataFrame
from xfeat.utils import analyze_columns, cudf_is_available


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


class DuplicatedFeatureEliminator(SelectorMixin):
    """Remove duplicated features."""

    def __init__(self):
        """[summary]."""
        self._selected_cols = []

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        if cudf_is_available() and isinstance(input_df, cudf.DataFrame):
            self._selected_cols = (
                input_df.to_pandas()
                .T.drop_duplicates(keep="first")
                .index.values.tolist()
            )
        else:
            self._selected_cols = input_df.T.drop_duplicates(
                keep="first"
            ).index.values.tolist()
        return input_df[self._selected_cols]

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[self._selected_cols]


class ConstantFeatureEliminator(SelectorMixin):
    """Remove constant features."""

    def __init__(self):
        """[summary]."""
        self._selected_cols = []

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        num_cols, cat_cols = analyze_columns(input_df)

        constant_cols = []
        for col in input_df.columns:
            if col in num_cols:
                if input_df[col].std() > 0:
                    continue
                value_count = input_df[col].count()
                if value_count == len(input_df) or value_count == 0:
                    constant_cols.append(col)

            elif col in cat_cols:
                value_count = input_df[col].count()
                if input_df[col].unique().shape[0] == 1 or value_count == 0:
                    constant_cols.append(col)

            else:
                # All nan values, like as [np.nan, np.nan, np.nan, np.nan, ...]
                constant_cols.append(col)

        self._selected_cols = [
            col for col in input_df.columns if col not in constant_cols
        ]

        return input_df[self._selected_cols]

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[self._selected_cols]
