import pytest
import pandas as pd
import numpy as np

from xfeat.selector._filter import (
    ChiSquareKBest,
    ANOVAClassifKBest,
    ANOVARegressionKBest,
    MutualInfoClassifKBest,
    SpearmanCorrelationEliminator,
)
from xfeat.utils import cudf_is_available


try:
    import cudf  # NOQA
except ImportError:
    cudf = None


@pytest.fixture
def dataframes():
    df = pd.DataFrame({"target": [1, 0, 0]})
    for col in range(100):
        df.loc[:, "col{}".format(col)] = np.array([1, 2, 3])

    if cudf_is_available():
        df_cuda = cudf.from_pandas(df)
        return [df, df_cuda]
    else:
        return [df]


def test_spearman_correlation_feature_eliminator(dataframes):
    for df in dataframes:
        selector = SpearmanCorrelationEliminator(threshold=0.5)
        selector.fit_transform(df)
        assert len(selector.get_selected_cols()) == 1


def test_chisquare_feature_selector(dataframes):
    for df in dataframes:
        selector = ChiSquareKBest(target_col="target", k=95)
        selector.fit_transform(df)
        assert len(selector.get_selected_cols()) == 95

        selector = ANOVAClassifKBest(target_col="target", k=95)
        selector.fit_transform(df)
        assert len(selector.get_selected_cols()) == 95

        selector = ANOVARegressionKBest(target_col="target", k=95)
        selector.fit_transform(df)
        assert len(selector.get_selected_cols()) == 95

        selector = MutualInfoClassifKBest(target_col="target", k=95)
        selector.fit_transform(df)
        assert len(selector.get_selected_cols()) == 95

        selector.reset_k(40)
        assert len(selector.get_selected_cols()) == 40
