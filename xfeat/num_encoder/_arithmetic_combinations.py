"""Module for ArithmeticCombinations class."""
from itertools import combinations

import pandas as pd

from xfeat.types import XDataFrame
from xfeat.base import TransformerMixin


class ArithmeticCombinations(TransformerMixin):
    """Calculate features by arithmetic combinations of numerical columns.

    Example:
        ::
            >>> import pandas as pd
            >>> from xfeat.num_encoder import ArithmeticCombinations
            >>> df = pd.DataFrame({
                "col1": [1, 1, 1, 1],
                "col2": [2, 4, 6, 8],
                "col3": [2, 4, 8, 16],
            })
            >>> encoder = ArithmeticCombinations()
            >>> encoder.fit_transform(df)
               col1  col2  col3  col1col2_plus  col1col3_plus  col2col3_plus
            0     1     2     2              3              3              4
            1     1     4     4              5              5              8
            2     1     6     8              7              9             14
            3     1     8    16              9             17             24
            >>> encoder = ArithmeticCombinations(
                output_suffix="", drop_origin=True
            )
            >>> encoder.fit_transform(df)
               col1col2  col1col3  col2col3
            0         3         3         4
            1         5         5         8
            2         7         9        14
            3         9        17        24
    """

    def __init__(
        self,
        input_cols=None,
        include_cols=None,
        exclude_cols=None,
        operator="+",
        output_prefix="",
        output_suffix="_combi",
        drop_origin=False,
        r=2,
    ):
        self._input_cols = input_cols or []
        self._include_cols = include_cols or []
        self._exclude_cols = exclude_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._r = r
        self._operator = operator
        self._drop_origin = drop_origin

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        new_df = input_df.copy()

        if not self._input_cols:
            self._input_cols = [
                col
                for col in new_df.columns.tolist()
                if (col not in self._exclude_cols)
            ]

        return self.transform(new_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        new_df = input_df.copy()
        generated_cols = []

        n_fixed_cols = len(self._include_cols)

        for cols_pairs in combinations(self._input_cols, r=self._r - n_fixed_cols):
            fixed_cols_str = "".join(self._include_cols)
            pairs_cols_str = "".join(cols_pairs)
            new_col = (
                self._output_prefix
                + fixed_cols_str
                + pairs_cols_str
                + self._output_suffix
            )
            generated_cols.append(new_col)

            concat_cols = self._include_cols + list(cols_pairs)
            new_ser = None

            for col in concat_cols:
                if new_ser is None:
                    new_ser = new_df[col].copy()
                else:
                    if self._operator == "+":
                        new_ser = new_ser + new_df[col]
                    elif self._operator == "-":
                        new_ser = new_ser - new_df[col]
                    elif self._operator == "*":
                        new_ser = new_ser * new_df[col]
                    elif self._operator == "/":
                        new_ser = new_ser / new_df[col]
                    elif self._operator == "%":
                        new_ser = new_ser % new_df[col]
                    else:
                        raise RuntimeError("Unknown operator is used.")

            new_df[new_col] = new_ser

        if self._drop_origin:
            return new_df[generated_cols]

        return new_df
