"""SelectNumerical encoders."""
from typing import List, Optional

import pandas as pd

from xfeat.utils import analyze_columns
from xfeat.base import TransformerMixin
from xfeat.types import XDataFrame


class SelectNumerical(TransformerMixin):
    """Select numerical columns.

    Args:
        use_cols (optional): [description]. Defaults to [].

        exclude_cols (Optional[List[str]]):
            Exclude column names.

    Example:
        ::

            >>> import pandas as pd
            >>> from xfeat.num_encoder import SelectNumerical
            >>> encoder = SelectNumerical()
            >>> encoder.fit_transform(pd.DataFrame({"col1": [1], "col2": ["b"]}))
              col1
            0    1
    """

    def __init__(
            self,
            use_cols=None,
            exclude_cols: Optional[List[str]] = None,
    ):
        """[summary]."""
        self._use_cols = use_cols or []
        self._exclude_cols = exclude_cols or []
        self._selected: List[str] = []

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            pd.DataFrame : Output data frame.
        """
        num_cols, cat_cols = analyze_columns(input_df)

        if self._use_cols:
            cat_cols = [col for col in cat_cols if col in self._use_cols]
        self._selected = num_cols

        if self._exclude_cols:
            for col in self._exclude_cols:
                self._selected.remove(col)

        return self.transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return input_df[self._selected].copy()
