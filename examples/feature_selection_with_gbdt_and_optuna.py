"""
This example uses UCI ML California Housing dataset, which is a
regression dataset including 20k samples.

    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of
    Information and Computer Science.
"""
from functools import partial

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna

from xfeat import ArithmeticCombinations, Pipeline
from xfeat import GBDTFeatureExplorer


def main():
    data = fetch_california_housing()
    df = pd.DataFrame(
        data=data.data,
        columns=data.feature_names)

    print("Before adding interaction features:")
    evaluate_dataframe(df, data.target)

    print("After adding interaction features:")
    df = feature_engineering(df)
    evaluate_dataframe(df, data.target)

    print("After applying GBDTFeatureSelector:")
    df = feature_selection(df, data.target)
    evaluate_dataframe(df, data.target)


def feature_engineering(df):
    cols = df.columns.tolist()

    encoder = Pipeline([
        ArithmeticCombinations(input_cols=cols,
                               drop_origin=False,
                               operator="+",
                               r=2,
                               output_suffix="_plus"),
        ArithmeticCombinations(input_cols=cols,
                               drop_origin=False,
                               operator="*",
                               r=2,
                               output_suffix="_mul"),
        ArithmeticCombinations(input_cols=cols,
                               drop_origin=False,
                               operator="-",
                               r=2,
                               output_suffix="_minus"),
        ArithmeticCombinations(input_cols=cols,
                               drop_origin=False,
                               operator="+",
                               r=3,
                               output_suffix="_plus"),
    ])
    return encoder.fit_transform(df)


def objective(df, selector, trial):
    selector.set_trial(trial)
    selector.fit(df)
    input_cols = selector.get_selected_cols()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }

    # Evaluate with selected columns
    train_set = lgb.Dataset(df[input_cols], label=df["target"])
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    return rmsle_score


def feature_selection(df, y):
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)

    df["target"] = np.log1p(y)
    df_train, _ = train_test_split(df, test_size=0.5, random_state=1)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    fit_params = {
        "num_boost_round": 100,
    }
    selector = GBDTFeatureExplorer(input_cols=input_cols,
                                   target_col="target",
                                   fit_once=True,
                                   threshold_range=(0.6, 1.0),
                                   lgbm_params=params,
                                   lgbm_fit_kwargs=fit_params)

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, df_train, selector), n_trials=20)

    selector.from_trial(study.best_trial)
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")

    return df[selected_cols]


def evaluate_dataframe(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df.values, y,
                                                        test_size=0.5,
                                                        random_state=1)
    y_train = np.log1p(y_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    print(f" - CV RMSEL: {rmsle_score:.6f}")

    booster = lgb.train(params, train_set, num_boost_round=100)
    y_pred = booster.predict(X_test)
    test_rmsle_score = rmse(np.log1p(y_test), y_pred)
    print(f" - test RMSEL: {test_rmsle_score:.6f}")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


if __name__ == "__main__":
    main()
