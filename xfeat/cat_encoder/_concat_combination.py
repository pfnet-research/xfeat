from itertools import combinations
from typing import List, Optional

import pandas as pd

from xfeat.types import XDataFrame
from xfeat.base import TransformerMixin


class ConcatCombination(TransformerMixin):
    """Generate combination of string columns.

    Example:
        ::

            >>> import pandas as pd
            >>> from xfeat import ConcatCombination
            >>> df = pd.DataFrame({
              "col1": ["a", "b"],
              "col2": ["@", "%"],
              "col3": ["X", "Y"]
            })
            >>> encoder = ConcatCombination()
            >>> encoder.fit_transform(df)
              col1 col2 col3 col1col2_combi col1col3_combi col2col3_combi
            0    a    @    X             a@             aX             @X
            1    b    %    Y             b%             bY             %Y

            >>> encoder = ConcatCombination(output_suffix="", drop_origin=True)
            >>> encoder.fit_transform(df)
              col1col2 col1col3 col2col3
            0       a@       aX       @X
            1       b%       bY       %Y

            >>> encoder = ConcatCombination(output_suffix="", drop_origin=True, r=3)
            >>> encoder.fit_transform(df)
              col1col2col3
            0          a@X
            1          b%Y

    Args:
        input_cols (Optional[List[str]]):
            Input column names. The default uses all columns of the input data frame.

        include_cols (Optional[List[str]]):
            Columns of the input data frame that are passed on to the output data frame.
            Defaults: None.

        output_prefix (str):
            Prefix of output column name. Defaults: `""`.

        output_suffix (str):
            Suffix of output column name. Defaults: `"_combi"`.

        drop_origin (bool):
            Drop the original column names. Defaults: `False`.

        fillna (str):
            To concatenate the string columns, the missing values are replaced with the
            string value `fillna`. Defaults: `"_NaN_"`.

        r (int):
            Length of combinations. Default: `2`.
    """

    def __init__(
        self,
        input_cols: Optional[List[str]] = None,
        include_cols: Optional[List[str]] = None,
        output_prefix: str = "",
        output_suffix: str = "_combi",
        drop_origin: bool = False,
        fillna: str = "_NaN_",
        r: int = 2,
    ):
        self._input_cols = input_cols or []
        self._include_cols = include_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._r = r
        self._fillna = fillna
        self._drop_origin = drop_origin

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.

        Returns:
            XDataFrame : Output data frame.
        """
        input_cols = self._input_cols

        if not input_cols:
            self._input_cols = [
                col
                for col in input_df.columns.tolist()
                if col not in self._include_cols
            ]

        return self.transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.

        Returns:
            XDataFrame : Output data frame.
        """
        cols = []

        n_fixed_cols = len(self._include_cols)
        df = input_df.copy()

        for cols_pairs in combinations(self._input_cols, r=self._r - n_fixed_cols):
            fixed_cols_str = "".join(self._include_cols)
            pairs_cols_str = "".join(cols_pairs)
            new_col = (
                self._output_prefix
                + fixed_cols_str
                + pairs_cols_str
                + self._output_suffix
            )
            cols.append(new_col)

            concat_cols = self._include_cols + list(cols_pairs)
            new_ser = None
            for col in concat_cols:
                if new_ser is None:
                    new_ser = df[col].fillna(self._fillna).copy()
                else:
                    new_ser = new_ser + df[col].fillna(self._fillna)

            df[new_col] = new_ser

        if self._drop_origin:
            return df[cols]

        return df
