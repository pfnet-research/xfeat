import pandas as pd

from xfeat.selector import GBDTFeatureSelector


def test_gbdt_feature_selector():
    cols = ["col1", "col2", "col3"]

    selector = GBDTFeatureSelector(input_cols=cols, target_col="target", threshold=0.5)
    assert selector  # TODO
