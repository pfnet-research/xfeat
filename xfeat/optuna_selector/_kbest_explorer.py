from typing import Union, List

import optuna

from xfeat.base import OptunaSelectorMixin
from xfeat.selector._filter import BaseSelectorKBest
from xfeat.types import XDataFrame


class KBestThresholdExplorer(OptunaSelectorMixin):
    """[summary].

    Args:
        selector (BaseSelectorKBest): [description].
        kbest_search_range (list[int] or str, optional):
          [description]. Defaults to 'auto'.
    """

    def __init__(
        self,
        selector: BaseSelectorKBest,
        kbest_search_range: Union[List[int], str] = "auto",
    ):
        self._selector = selector
        self._trial = None
        self.kbest_search_range = kbest_search_range

        self._search_space = {
            "KBestThresholdExplorer.k": {
                "dist_class": "optuna.distributions.DiscreteUniformDistribution",
                "dist_args": [],
            },
        }

    def set_trial(self, trial):
        """[summary].

        Args:
            trial : [description].
        """
        self._trial = trial

    def _set_params(self):
        if self.kbest_search_range == "auto":
            n_cols = len(self._selector._input_cols)
            self._search_space["KBestThresholdExplorer.k"]["dist_args"] = [
                min(1, n_cols),
                n_cols,
                1,
            ]
        else:
            self._search_space["KBestThresholdExplorer.k"][
                "dist_args"
            ] = self.kbest_search_range

        param_name = "KBestThresholdExplorer.k"
        dist_info = self._search_space[param_name]
        class_ = self._dynamic_load(dist_info["dist_class"])
        dist = class_(*dist_info["dist_args"])

        self._selector.reset_k(int(self._trial._suggest(param_name, dist)))

    def from_trial(self, trial: optuna.trial.FrozenTrial):
        """[summary].

        Args:
            trial (optuna.trial.FrozenTrial): [description].
        """
        self._trial = trial
        if "KBestThresholdExplorer.k" in trial.params:
            self._selector.reset_k(int(trial.params["KBestThresholdExplorer.k"]))

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        Returns:
            XDataFrame : [description].
        """
        self._selector._lazy_setup_input_cols(input_df.columns.tolist())

        if self._trial is None:
            raise RuntimeError("Call set_trial(trial) before fit_transform().")

        if not isinstance(self._trial, optuna.trial.FrozenTrial):
            self._set_params()

        return self._selector.fit_transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        Returns:
            XDataFrame : [description].
        """
        if self._trial is None:
            raise RuntimeError("Call set_trial(trial) before transform().")

        return self._selector.transform(input_df)

    def get_selected_cols(self):
        """[summary].

        Returns:
            Any : [description].
        """
        return self._selector.get_selected_cols()
