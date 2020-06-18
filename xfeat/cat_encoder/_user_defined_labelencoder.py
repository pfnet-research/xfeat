"""User defined label encoder."""
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from xfeat.base import TransformerMixin
from xfeat.types import XDataFrame


class UserDefinedLabelEncoder(TransformerMixin):
    """Encode labels with user-defined values.

    Args:
        label_mapping (Dict[str, int]):
            User-defined mapping of label encoding.

        input_cols (Optional[List[str]]):
            Input column names. The default uses all columns of the input data frame.

        output_prefix (str):
            Prefix of output column name. Defaults: `""`.

        output_suffix (str):
            Suffix of output column name. Defaults: `"_le"`.

        unseen (str):
            Select methods for handling unseen values. `minus_one` or `n_unique`.
            By default `-1` is assigned for unseen values and missing (NaN) values.
    """

    def __init__(
        self,
        label_mapping: Dict[str, int],
        input_cols: Optional[List[str]] = None,
        output_prefix: str = "",
        output_suffix: str = "_le",
        unseen: str = "minus_one",
    ):
        self._label_mapping = label_mapping
        self._input_cols = input_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._unseen = unseen

        self._labels: Dict[str, List[Any]] = {}
        self._uniques: Dict[str, pd.Index] = {}

    def fit(self, input_df: XDataFrame) -> None:
        """Fit to data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        input_cols = self._input_cols
        if not input_cols:
            input_cols = input_df.columns.tolist()

        for col in input_cols:
            labels = list(self._label_mapping.values())
            uniques = pd.Index(list(self._label_mapping.keys()))
            self._labels[col] = labels
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

        for col in input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            X = self._uniques[col].get_indexer(new_df[col])

            if self._unseen == "n_unique":
                missing_values = new_df[col].isna()
                unseen_values = np.invert(new_df[col].isin(self._uniques[col]))
                unseen_mask = np.bitwise_xor(missing_values, unseen_values)
                X[unseen_mask] = len(self._uniques[col])

            missing_values = new_df[col].isna()
            unseen_values = np.invert(new_df[col].isin(self._uniques[col]))
            unseen_mask = np.bitwise_xor(missing_values, unseen_values)
            X[~unseen_mask] = np.array(self._labels[col])[X[~unseen_mask]]

            new_df[out_col] = X

        return new_df
