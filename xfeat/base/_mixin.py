import importlib
from typing import Any, List, Dict

import optuna

from xfeat.types import XDataFrame


class TransformerMixin:
    """Mixin class for `xfeat.encoder`."""

    def fit(self, input_df: XDataFrame) -> None:
        """Fit to data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        raise NotImplementedError("Not implemented yet.")

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        raise NotImplementedError("Not implemented yet.")

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        self.fit(input_df)
        return self.transform(input_df)


class OptunaSelectorMixin:
    """Mixin class for `xfeat.optuna_selector`."""

    def __init__(self):
        """Define search space in `self.get_search_space`."""
        self._search_space = {}

    def _dynamic_load(self, model_class_fqn):
        module_name = ".".join(model_class_fqn.split(".")[:-1])
        class_name = model_class_fqn.split(".")[-1]
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        return cls

    def fit(self, input_df: XDataFrame) -> None:
        """Fit to data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        raise NotImplementedError("Not implemented yet.")

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        raise NotImplementedError("Not implemented yet.")

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        self.fit(input_df)
        return self.transform(input_df)

    def set_trial(self, trial):
        """Set trial object.

        Args:
            trial : trial object from Optuna.
        """
        raise NotImplementedError("Not implemented yet.")

    def from_trial(self, trial: optuna.trial.FrozenTrial):
        """Load parameters from trial.

        Args:
            trial (optuna.trial.FrozenTrial): [description].
        """
        raise NotImplementedError("Not implemented yet.")

    def get_gridsearch_space(self) -> Dict[str, Any]:
        """Get search space used for `optuna.samplers.GridSampler`."""
        raise NotImplementedError("Not implemented yet.")

    def gridsearch_space_size(self) -> int:
        """Return the total number of trial for gridseearch."""
        raise NotImplementedError("Not implemented yet.")

    def get_selected_cols(self) -> List[str]:
        """Get selected columns."""
        raise NotImplementedError("Not implemented yet.")


class SelectorMixin:
    """Mixin class for `xfeat.selector`."""

    def fit(self, input_df: XDataFrame) -> None:
        """Fit to data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        """
        raise NotImplementedError("Not implemented yet.")

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """Transform data frame.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        raise NotImplementedError("Not implemented yet.")

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """Fit to data frame, then transform it.

        Args:
            input_df (XDataFrame): Input data frame.
        Returns:
            XDataFrame : Output data frame.
        """
        self.fit(input_df)
        return self.transform(input_df)
