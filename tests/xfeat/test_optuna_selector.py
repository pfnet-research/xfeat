from functools import partial

import optuna
import pandas as pd

from xfeat.selector._filter import ChiSquareKBest
from xfeat.optuna_selector._kbest_explorer import KBestThresholdExplorer
from xfeat.optuna_selector import GBDTFeatureExplorer


def test_kbest_threshold_explorer():
    cols = ["col1", "col2", "col3"]
    selector = KBestThresholdExplorer(
        ChiSquareKBest(input_cols=cols, target_col="target")
    )

    assert selector  # TODO


def test_gbdt_feature_explorer():
    cols = ["col1", "col2", "col3"]
    selector = GBDTFeatureExplorer(
        input_cols=cols, target_col="target", threshold_range=(0.1, 0.9)
    )

    assert selector  # TODO
