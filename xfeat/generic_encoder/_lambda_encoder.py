"""Encoders for categorial and numerical columns."""
import pandas as pd

from xfeat.types import XDataFrame
from xfeat.base import TransformerMixin
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


class LambdaEncoder(TransformerMixin):
    """Encoder with user-defined function.

    Args:
        lambda_func : User-defined function.
        input_cols (optional): [description]. Defaults to [].
        exclude_cols (optional): [description]. Defaults to [].
        output_prefix (optional): [description]. Defaults to ''.
        output_suffix (optional): [description]. Defaults to '_lmd'.
        fillna (optional): [description]. Defaults to 'NaN'.
    """

    def __init__(
        self,
        lambda_func,
        input_cols=[],
        exclude_cols=[],
        output_prefix="",
        output_suffix="_lmd",
        drop_origin=False,
        fillna="NaN",
    ):
        self._lambda_func = lambda_func
        self._input_cols = input_cols
        self._exclude_cols = exclude_cols
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._fillna = fillna
        self._drop_origin = drop_origin

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        return self.transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        if isinstance(input_df, pd.DataFrame):
            new_df = input_df.copy()
        elif cudf_is_available() and isinstance(input_df, cudf.DataFrame):
            new_df = input_df.to_pandas()
        else:
            raise RuntimeError("Unexpected data type: {}".format(type(input_df)))
        generated_cols = []

        input_cols = self._input_cols
        if not input_cols:
            input_cols = new_df.columns.tolist()
        if len(self._exclude_cols) > 0:
            input_cols = [col for col in input_cols if col not in self._exclude_cols]

        for col in input_cols:
            new_col = self._output_prefix + col + self._output_suffix
            if self._fillna is not None:
                new_df[new_col] = (
                    new_df[col].fillna(self._fillna).apply(self._lambda_func)
                )
            else:
                new_df[new_col] = new_df[col].apply(self._lambda_func)

            generated_cols.append(new_col)

        if cudf_is_available() and isinstance(input_df, cudf.DataFrame):
            new_df = cudf.from_pandas(new_df)

        if self._drop_origin:
            return new_df[generated_cols]

        return new_df
