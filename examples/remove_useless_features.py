"""
This example uses UCI ML Breast Cancer Wisconsin (Diagnostic) dataset, which is a
classic and very easy binary classification dataset.

    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of
    Information and Computer Science.
"""
import pandas as pd

from sklearn.datasets import load_breast_cancer
from xfeat.pipeline import Pipeline
from xfeat.selector import DuplicatedFeatureEliminator
from xfeat.selector import ConstantFeatureEliminator
from xfeat.selector import SpearmanCorrelationEliminator
from xfeat.utils import compress_df


def get_feature_selector():
    return Pipeline(
        [
            DuplicatedFeatureEliminator(),
            ConstantFeatureEliminator(),
            SpearmanCorrelationEliminator(threshold=0.8),
        ]
    )


def main():
    data = load_breast_cancer()
    df = compress_df(pd.DataFrame(data=data.data, columns=data.feature_names))

    selector = get_feature_selector()
    df_reduced = selector.fit_transform(df)
    print("Selected columns: {}".format(df_reduced.columns.tolist()))


if __name__ == "__main__":
    main()
