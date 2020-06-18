"""
This example uses UCI ML Breast Cancer Wisconsin (Diagnostic) dataset, which is a
classic and very easy binary classification dataset.

    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of
    Information and Computer Science.
"""
from functools import partial

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from ml_metrics import auc

from xfeat.pipeline import Pipeline
from xfeat.num_encoder import SelectNumerical
from xfeat.selector import ChiSquareKBest
from xfeat.optuna_selector import KBestThresholdExplorer


def _load_dataset():
    breast_cancer_dataset = load_breast_cancer()
    df = pd.DataFrame(
        breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names
    )
    df.loc[:, "target"] = breast_cancer_dataset.target
    return df


def objective(df, selector, trial):
    selector.set_trial(trial)

    # NOTE: Use validation set for practical usage.
    df_trn, df_tst = train_test_split(df, test_size=0.5)
    X_trn = selector.fit_transform(df_trn).values
    X_tst = selector.transform(df_tst).values

    model = LogisticRegression(solver="lbfgs", max_iter=10000)
    model.fit(X_trn, df_trn["target"])
    y_pred = model.predict_proba(X_tst)[:, 1]

    score = auc(df_tst["target"].values, y_pred)
    return score


def main():
    df = _load_dataset()
    selector = Pipeline(
        [
            SelectNumerical(),
            KBestThresholdExplorer(ChiSquareKBest(target_col="target")),
        ]
    )

    # Optimize with Optuna!
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, df, selector), n_trials=20)

    # Load from the best trial
    selector.from_trial(study.best_trial)
    print("Selected cols:", selector.get_selected_cols())


if __name__ == "__main__":
    main()
