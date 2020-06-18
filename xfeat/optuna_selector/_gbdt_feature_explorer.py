from typing import Tuple, Any, Dict, List

import optuna
import pandas as pd

from xfeat.base import OptunaSelectorMixin
from xfeat.selector import GBDTFeatureSelector
from xfeat.types import XDataFrame


_DEFAULT_THRESHOLD = 0.9
_LGBM_DEFAULT_PARAMS = {
    "objective": "regression",
}

_LGBM_DEFAULT_FIT_KWARGS = {
    "num_boost_round": 100,
}


class GBDTFeatureExplorer(GBDTFeatureSelector, OptunaSelectorMixin):
    """[summary].

    Args:
        input_cols : [description].
        target_col (optional): [description]. Defaults to 'target'.
        fit_once (bool): Defauls to `True`.
        threshold_range (optional): [description]. Defaults to (0.1, 0.9).
        lgbm_params (optional): [description]. Defaults to _LGBM_DEFAULT_PARAMS.
        lgbm_fit_kwargs (optional):
            [description]. Defaults to _LGBM_DEFAULT_FIT_KWARGS.
    """

    def __init__(
        self,
        input_cols: List[str],
        target_col: str = "target",
        fit_once: bool = True,
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        lgbm_params: Dict[str, Any] = _LGBM_DEFAULT_PARAMS,
        lgbm_fit_kwargs: Dict[str, Any] = _LGBM_DEFAULT_FIT_KWARGS,
    ):
        self._input_cols = input_cols
        self._target_col = target_col
        self._fit_once = fit_once

        # default threshold is used when `set_trial()` is not called.
        self._threshold = _DEFAULT_THRESHOLD

        # TODO(smly): Add validator for lgbm related args
        self._lgbm_params = lgbm_params
        self._lgbm_fit_kwargs = lgbm_fit_kwargs

        self._booster = None
        self._selected_cols = None
        self.select_cols_count = int(self._threshold * len(self._input_cols))

        self._init_search_space(threshold_range)

    def _init_search_space(self, threshold_range):
        self._search_space = {
            "GBDTFeatureSelector.threshold": {
                "dist_class": "optuna.distributions.UniformDistribution",
                "dist_args": list(threshold_range),
            },
        }

    def set_trial(self, trial):
        """[summary].

        Args:
            trial : [description].
        """
        param_name = "GBDTFeatureSelector.threshold"
        dist_info = self._search_space[param_name]
        class_ = self._dynamic_load(dist_info["dist_class"])
        dist = class_(*dist_info["dist_args"])

        self._threshold = trial._suggest(param_name, dist)

    def from_trial(self, trial: optuna.trial.FrozenTrial):
        """[summary].

        Args:
            trial (optuna.trial.FrozenTrial): [description].
        """
        if "GBDTFeatureSelector.threshold" in trial.params:
            self._threshold = trial.params["GBDTFeatureSelector.threshold"]

    def fit(self, input_df: XDataFrame) -> None:
        if self._booster is not None and self._fit_once:
            self.select_cols_count = int(
                self._threshold * len(self._input_cols))

            if self.select_cols_count == 0:
                # TODO(smly): Make this clear. Use logger instad of print
                # function.
                print("threshold is too small.")

            feature_importance = pd.DataFrame(
                {"col": self._input_cols, "importance": self._booster.feature_importance()}
            )
            self._selected_cols = (
                feature_importance.sort_values(
                    by="importance", ascending=False) .head(
                    self.select_cols_count) .col.tolist())
        else:
            super().fit(input_df)

    def get_selected_cols(self):
        """[summary].

        Returns:
            Any : [description].
        """
        return self._selected_cols
