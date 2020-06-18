from logging import getLogger
from typing import List, Optional

import pandas as pd
import optuna

from xfeat.types import XDataFrame
from xfeat.base import TransformerMixin, OptunaSelectorMixin


logger = getLogger(__name__)


class Pipeline(TransformerMixin):
    """[summary].

    Args:
        transforms : [description].
    """

    def __init__(self, transforms):
        self._transforms = transforms

    def fit(self, input_df: XDataFrame) -> None:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        """
        raise RuntimeError("Pipeline doesnt support fit(). Use fit_transform().")

    def from_trial(self, trial: optuna.trial.FrozenTrial):
        """[summary].

        Args:
            trial (optuna.trial.FrozenTrial): [description].
        """
        for transform in self._transforms:
            if isinstance(transform, OptunaSelectorMixin):
                transform.from_trial(trial)

    def set_trial(self, trial):
        """[summary].

        Args:
            trial : [description].
        """
        for transform in self._transforms:
            if isinstance(transform, OptunaSelectorMixin):
                transform.set_trial(trial)

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        """
        for transform in self._transforms:
            input_df = transform.fit_transform(input_df)
        return input_df

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        """[summary].

        Args:
            input_df (XDataFrame): [description].
        Returns:
            XDataFrame : [description].
        """
        for transform in self._transforms:
            input_df = transform.transform(input_df)
        return input_df

    def get_selected_cols(self) -> Optional[List[str]]:
        """[summary].

        Returns:
            list[str] or None : [description].
        """
        for i, transform in enumerate(self._transforms, start=1):
            if isinstance(transform, OptunaSelectorMixin):
                if i == len(self._transforms):
                    logger.warning("Optuna selector is not last component.")

                return transform.get_selected_cols()

        return None
