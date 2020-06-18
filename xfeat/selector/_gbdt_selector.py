"""Module for GBDTFeatureSelector."""
from logging import getLogger
from typing import List

import lightgbm as lgb
import pandas as pd


from xfeat.base import SelectorMixin
from xfeat.types import XDataFrame


_LGBM_DEFAULT_PARAMS = {
    "objective": "regression",
}

_LGBM_DEFAULT_FIT_KWARGS = {
    "num_boost_round": 100,
}


logger = getLogger(__name__)


class GBDTFeatureSelector(SelectorMixin):
    """Feature selector by using LightGBM model.

    Args:
        input_cols (optional): [description]. Defaults to [].
        target_col (optional): [description]. Defaults to 'target'.
        threshold (optional): [description]. Defaults to 0.9.
        lgbm_params (optional): [description]. Defaults to _LGBM_DEFAULT_PARAMS.
        lgbm_fit_kwargs (optional):
          [description]. Defaults to _LGBM_DEFAULT_FIT_KWARGS.
    """

    def __init__(
        self,
        input_cols=None,
        target_col="target",
        threshold=0.9,
        lgbm_params=_LGBM_DEFAULT_PARAMS,
        lgbm_fit_kwargs=_LGBM_DEFAULT_FIT_KWARGS,
    ):
        """[summary]."""
        self._input_cols = input_cols
        self._target_col = target_col
        self._threshold = threshold

        # TODO(smly): Add validator for lgbm related args
        self._lgbm_params = lgbm_params
        self._lgbm_fit_kwargs = lgbm_fit_kwargs

        self._booster = None
        self._selected_cols = None
        self.select_cols_count = None

    def fit(self, input_df: XDataFrame) -> None:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        """
        if not self._input_cols:
            cols = input_df.columns.tolist()
            cols.remove(self._target_col)
            self._input_cols = cols

        self.select_cols_count = int(self._threshold * len(self._input_cols))

        if self.select_cols_count == 0:
            # TODO(smly): Make this message better.
            logger.warning("threshold is too small.")

        train_data = lgb.Dataset(input_df[self._input_cols], input_df[self._target_col])
        self._booster = lgb.train(
            self._lgbm_params, train_data, **self._lgbm_fit_kwargs
        )

        feature_importance = pd.DataFrame(
            {"col": self._input_cols, "importance": self._booster.feature_importance()}
        )
        self._selected_cols = (
            feature_importance.sort_values(by="importance", ascending=False)
            .head(self.select_cols_count)
            .col.tolist()
        )

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        """
        if self._booster is None or self.select_cols_count is None:
            raise RuntimeError("Call fit() before transform().")

        return input_df[self._selected_cols]

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        Returns:
            XDataFrame : [description].
        """
        self.fit(input_df)
        return self.transform(input_df)

    def get_selected_cols(self) -> List[str]:
        """[summary].

        Returns:
            Any : [description].
        """
        return self._selected_cols
