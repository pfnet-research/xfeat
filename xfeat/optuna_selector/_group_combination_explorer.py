"""Module for GroupCombinationExplorer."""
from typing import Any, List, Dict

import pandas as pd
import optuna

from xfeat.base import OptunaSelectorMixin
from xfeat.types import XDataFrame


class GroupCombinationExplorer(OptunaSelectorMixin):
    """Find best combination of column groups.

    Example:
        ::

            >>> from xfeat.optuna_selector import GroupCombinationExplorer
            >>> selector = GroupCombinationExplorer(
            >>>     base_cols=["v1", "v2", "v3", "v4"]
            >>>     list_groups={
            >>>         "group1": ["v5", "v6"],
            >>>         "group2": ["v7", "v8"],
            >>>     },
            >>> )
            >>> def objective(df, selector, trial):
            >>>     selector.set_trial(trial)
            >>>     df_select = selector.fit_transform(df)
            >>>     # snip
            >>>     return score

    """

    def __init__(self, base_cols: List[str], list_groups: Dict[str, List[str]]):
        """[summary]."""
        self._base_cols = base_cols
        self._list_groups = list_groups

        # TODO: validate `base_cols` and `list_groups` for avoiding runtime errors.

        self._init_search_space(list_groups)

        # Set default parameters
        self._select_flags = {}
        for group_label in list_groups.keys():
            self._select_flags[group_label] = True

    def _init_search_space(self, list_groups: Dict[str, List[str]]):
        self._search_space = {}

        for group_label in list_groups.keys():
            self._search_space[
                "GroupCombinationExplorer.flag_{}".format(group_label)
            ] = {
                "dist_class": "optuna.distributions.CategoricalDistribution",
                "dist_args": [[True, False]],
            }

    def set_trial(self, trial):
        """Set trial object.

        Args:
            trial : Optuna's trial object.
        """
        param_names = [
            param_name
            for param_name in self._search_space.keys()
            if param_name.startswith("GroupCombinationExplorer")
        ]

        for param_name in param_names:
            dist_info = self._search_space[param_name]
            class_ = self._dynamic_load(dist_info["dist_class"])
            dist = class_(*dist_info["dist_args"])

            group_label = param_name[len("GroupCombinationExplorer.flag_") :]
            self._select_flags[group_label] = trial._suggest(param_name, dist)

    def from_trial(self, trial: optuna.trial.FrozenTrial):
        """Load parameters from trial.

        Args:
            trial (optuna.trial.FrozenTrial): Optuna's frozen trial.
        """
        param_names = [
            param_name
            for param_name in self._search_space.keys()
            if param_name.startswith("GroupCombinationExplorer")
        ]

        for param_name in param_names:
            if param_name in trial.params:
                group_label = param_name[len("GroupCombinationExplorer.flag_") :]
                self._select_flags[group_label] = trial.params[param_name]
            else:
                # TODO(smly): Make this better.
                raise RuntimeError("Invalid paramter found.")

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        """
        return self.transform(input_df)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        Returns:
            XDataFrame : [description].
        """
        use_cols = self._base_cols.copy()
        for group_label in self._list_groups:
            if self._select_flags[group_label]:
                for grp_col in self._list_groups[group_label]:
                    # TODO(smly): Validate duplications and remove here)
                    assert grp_col not in use_cols

                    use_cols.append(grp_col)

        return input_df[use_cols]

    def get_selected_cols(self) -> List[str]:
        """[summary].

        Returns:
            List[str]: [description].
        """
        selected_cols = self._base_cols.copy()

        for group_label in self._list_groups:
            if self._select_flags[group_label]:
                selected_cols += self._list_groups[group_label]

        return selected_cols

    def get_selected_groups(self) -> Dict[str, bool]:
        """[summary]."""
        return self._select_flags

    def get_search_space(self) -> Dict[str, Any]:
        """[summary].

        Returns:
            Dict[str, Any] : [description].
        """
        return {
            "GroupCombinationExplorer.flag_{}".format(group_label): [True, False]
            for group_label in self._list_groups.keys()
        }

    def gridsearch_space_size(self) -> int:
        """[summary].

        Returns:
            int : [description].
        """
        return 2 ** len(self._list_groups.keys())
