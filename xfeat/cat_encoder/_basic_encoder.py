"""Basic categorical encoders."""
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from xfeat.utils import analyze_columns
from xfeat.base import TransformerMixin
from xfeat.types import XDataFrame


class LabelEncoder(TransformerMixin):
    """Encode labels with numerical values between `0` and `n_unique - 1`.

    Example:
        `LabelEncoder` can be used to encode `string` labels.

        ::

            >>> import pandas as pd
            >>> from xfeat import LabelEncoder
            >>> encoder = LabelEncoder()
            >>> encoder.fit_transform(pd.DataFrame({"col": ["a", "b", "b"]}))
              col  col_le
            0   a       0
            1   b       1
            2   b       1
            >>> encoder.transform(pd.DataFrame({"col": ["b"]}))
              col  col_le
            0   b       1

    Args:
        input_cols (Optional[List[str]]):
            Input column names. The default uses all columns of the input data frame are
            used.

        exclude_cols (Optional[List[str]]):
            Exclude column names.

        output_prefix (str):
            Prefix of output column name. Defaults: `""`.

        output_suffix (str):
            Suffix of output column name. Defaults: `"_le"`.

        sort_category (bool):
            Sort category before fitting. Defaults: `False`.

        unseen (str):
            Select methods for handling unseen values. `minus_one` or `n_unique`.
            By default `-1` is assigned for unseen values and missing (NaN) values.
    """

    def __init__(
        self,
        input_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        output_prefix: str = "",
        output_suffix: str = "_le",
        sort_category: bool = False,
        unseen: str = "minus_one",
    ):
        self._input_cols = input_cols or []
        self._exclude_cols = exclude_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._sort_category = sort_category
        self._unseen = unseen

        self._uniques: Dict[str, pd.Index] = {}

    def fit(self, input_df: XDataFrame) -> None:
        """Fit to data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        input_cols = self._input_cols
        if not input_cols:
            input_cols = input_df.columns.tolist()

        if self._exclude_cols:
            for col in self._exclude_cols:
                input_cols.remove(col)

        for col in input_cols:
            if isinstance(input_df[col], pd.Series):
                labels, uniques = input_df[col].factorize(sort=self._sort_category)
            else:
                labels, uniques = (
                    input_df[col].to_pandas().factorize(sort=self._sort_category)
                )

            self._uniques[col] = uniques

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        self.fit(input_df)
        return self.transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        new_df = input_df.copy()

        input_cols = self._input_cols
        if not input_cols:
            input_cols = new_df.columns.tolist()

        if self._exclude_cols:
            for col in self._exclude_cols:
                input_cols.remove(col)

        for col in input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            X = self._uniques[col].get_indexer(new_df[col])

            if self._unseen == "n_unique":
                missing_values = new_df[col].isna()
                unseen_values = np.invert(new_df[col].isin(self._uniques[col]))
                unseen_mask = np.bitwise_xor(missing_values, unseen_values)
                X[unseen_mask] = len(self._uniques[col])

            new_df[out_col] = X

        return new_df


class SelectCategorical(TransformerMixin):
    """Select categorical columns.

    Example:
        ::

            >>> import pandas as pd
            >>> from xfeat.cat_encoder import SelectCategorical
            >>> encoder = SelectCategorical()
            >>> encoder.fit_transform(pd.DataFrame({"col1": [1], "col2": ["b"]}))
              col2
            0    b

    Args:
        use_cols (Optional[List[str]]):
            Input column names. The default uses all columns of the input data frame.

        exclude_cols (Optional[List[str]]):
            Exclude column names.
    """

    def __init__(
            self,
            use_cols=None,
            exclude_cols: Optional[List[str]] = None,
    ):
        self._use_cols = use_cols or []
        self._exclude_cols = exclude_cols or []
        self._selected: List[str] = []

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.

        Returns:
            XDataFrame : Output data frame.
        """
        num_cols, cat_cols = analyze_columns(input_df)

        if self._use_cols:
            cat_cols = [col for col in cat_cols if col in self._use_cols]
        self._selected = cat_cols

        if self._exclude_cols:
            for col in self._exclude_cols:
                self._selected.remove(col)

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.

        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[self._selected].copy()
