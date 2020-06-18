from functools import partial

import optuna
from optuna.samplers import GridSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# from ml_metrics import accuracy

from xfeat.optuna_selector import GroupCombinationExplorer


def objective(df, selector, trial):
    selector.set_trial(trial)

    # NOTE: Use validation set for practical usage.
    df_trn, df_tst = train_test_split(df, test_size=0.5)
    X = selector.fit_transform(df).values
    X_tst = selector.transform(df_tst).values
    print(X.shape)

    model = LogisticRegression(solver="lbfgs", max_iter=10000)
    model.fit(X, df["target"])
    y_pred = model.predict_proba(X_tst)[:, 1]

    score = (df_tst["target"].values == y_pred).mean()
    return score


def main():
    selector = GroupCombinationExplorer(
        base_cols=["v1", "v2"], list_groups={"group1": ["v3", "v4"], "group2": ["v5"],},
    )

    df = pd.DataFrame(
        {
            "v1": [1, 1, 1, 1,],
            "v2": [2, 2, 2, 2,],
            "v3": [2, 2, 2, 2,],
            "v4": [2, 2, 2, 2,],
            "v5": [2, 2, 2, 2,],
            "target": [0, 0, 0, 1,],
        }
    )

    # Optimize with Optuna!
    study = optuna.create_study(
        direction="maximize", sampler=GridSampler(selector.get_search_space())
    )
    study.optimize(
        partial(objective, df, selector), n_trials=selector.gridsearch_space_size()
    )

    # Load from the best trial
    selector.from_trial(study.best_trial)
    print("Selected cols:", selector.get_selected_cols())


if __name__ == "__main__":
    main()
