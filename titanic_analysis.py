# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# TODO: better imputer!
# TODO: try ensemble.AdaBoostClassifier
# TODO: read about XGBoost
# TODO: features: encode only common things
# TODO: features: remove "is zero"
# TODO: why cross_val is 0.85, but test is 0.79?
# TODO: use shap to inspect
# -

# # titanic model

# %load_ext autoreload

# %autoreload 1
# %aimport titanic

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import optuna.visualization as ov
import pandas as pd
from sklearn import calibration, metrics, model_selection
from sklearn.model_selection import cross_validate, train_test_split
from xgboost import XGBClassifier

from titanic.model import build_model, build_preprocess, evaluate_model
from titanic.notebook import DataFrameDisplay, compare_col, create_axs

# -


SPLIT_SEED = 0

# ## load the data

input_dir = Path("/kaggle/input/titanic")
if not input_dir.exists():
    input_dir = Path("data")

kaggle_train_df = pd.read_csv(input_dir / "train.csv", index_col=0)
kaggle_test_df = pd.read_csv(input_dir / "test.csv", index_col=0)

kaggle_train_df.sample(5).sort_index()

# ## learn

# ### split

# +
# add validation for final estimation of the model
train_df, val_df = train_test_split(
    kaggle_train_df, test_size=100, stratify=kaggle_train_df.Survived, random_state=SPLIT_SEED
)
train_target = train_df.pop("Survived")
val_target = val_df.pop("Survived")

len(train_df), len(val_df)
# -


# ### find best model

# #### which features?

preprocess_parameters = list(inspect.signature(build_preprocess).parameters)
results_1 = DataFrameDisplay(["param"])
for param_name in ["base", *preprocess_parameters]:
    model = build_model(
        transformer=build_preprocess(**{p: p == param_name for p in preprocess_parameters}),
        classifier=XGBClassifier(),
    )
    result = evaluate_model(model, train_df, train_target)
    result["num_features"] = len(model[:-1].fit(train_df).get_feature_names_out())
    results_1.add_row(result, param_name)
assert results_1.df is not None


results_1.df.plot.scatter(x="num_features", y="accuracy_μ", yerr="accuracy_σ")

(results_1.df - results_1.df.loc["base"])[["fit_time", "accuracy_μ"]].sort_values(by="fit_time")


# #### which hyperparams?


# +
def objective(trial: optuna.Trial) -> float:
    model = build_model(
        transformer=build_preprocess(),
        classifier=XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 1, 300),
            max_depth=trial.suggest_int("max_depth", 1, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        ),
    )
    scores = cross_validate(model, train_df, train_target)
    score: float = scores["test_score"].mean()
    return score


objective_name = "accuracy"

study_name = f"XGBClassifier-params-with-cabin_even-split_{SPLIT_SEED}"
storage_name = f"sqlite:///cache/{study_name}.db"
study = optuna.create_study(
    study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True
)
n_trails = max((-len(study.trials) - 1) % 10 + 1, 100 - len(study.trials))
study.optimize(objective, n_trials=n_trails, n_jobs=-1)
# -

len(study.trials), study.best_params

study.trials_dataframe().set_index("number").tail()

ov.plot_optimization_history(study, target_name=objective_name)

ov.plot_parallel_coordinate(study, target_name=objective_name)

ov.plot_slice(study, target_name=objective_name)

axis = create_axs(3)
for col in study.trials_dataframe(multi_index=True)["params"]:
    (
        study.trials_dataframe()
        .rename(columns=lambda s: s.replace("params_", ""))
        .rename(columns={"value": objective_name})
    ).plot.scatter(x="number", c=objective_name, y=col, ax=next(axis))
plt.tight_layout()

ov.plot_param_importances(study, target_name=objective_name)

best_depth = study.best_params["max_depth"]
marker = dict(marker="o", ms=8, markerfacecolor="None", markeredgecolor="k", markeredgewidth=0.5)
max_depths = [best_depth - 1, best_depth, best_depth + 1]
study_df = study.trials_dataframe(multi_index=True)["params"].join(
    study.trials_dataframe()["value"].rename(objective_name)
)
n_cols = len(max_depths)
fig, axs = plt.subplots(1, n_cols, sharey=True, sharex=True, figsize=(6 * n_cols, 4))
axis = iter(axs)
for max_depth in max_depths:
    ax = study_df[study_df.max_depth == max_depth].plot.scatter(
        y="learning_rate",
        x="n_estimators",
        c="accuracy",
        cmap="turbo",
        logy=True,
        s=2,
        ax=next(axis),
        title=f"{max_depth=}",
    )
    if max_depth == study.best_params["max_depth"]:
        ax.plot(study.best_params["n_estimators"], study.best_params["learning_rate"], **marker)
fig.tight_layout()


# ### understand

# +
model = build_model(classifier=XGBClassifier(**study.best_params))
model.fit(train_df, train_target)

features_df = pd.Series(
    model[-1].feature_importances_, index=model[-1].feature_names_in_
).sort_values(ascending=False)
features_df.head(10).plot.bar()
# -

_fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(train_df, train_target, "PrefixName_Mr", model).plot.bar(
    ax=axs[0], ylim=(0, len(train_df))
)
compare_col(train_df, train_target, "PrefixName_Miss", model).plot.bar(
    ax=axs[1], ylim=(0, len(train_df))
)

_fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(train_df, train_target, "Fare").plot.bar(ax=axs[0])
compare_col(train_df, train_target, "Age").plot.bar(ax=axs[1])


# ### finalize_and_predict


# +
def predict(model, data):
    return pd.Series(model.predict(data), index=data.index, name="Survived")


def val_score(model):
    scores = {}
    scores["train_cv"] = cross_validate(model, train_df, train_target)["test_score"]
    scores["train_score"] = scores["train_cv"].mean()
    scores["train_score_median"] = pd.Series(scores["train_cv"]).median()
    model.fit(train_df, train_target)
    scores["val_score"] = model.score(val_df, val_target)
    for metric in [
        "accuracy",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
    ]:
        scorer = metrics.get_scorer(metric)
        scores[metric] = scorer(model, val_df, val_target)

    axis = create_axs(2, ax_size=(6, 4))
    metrics.ConfusionMatrixDisplay.from_estimator(
        model, val_df, val_target, display_labels=["ns", "s"], ax=next(axis)
    )
    metrics.DetCurveDisplay.from_estimator(model, val_df, val_target, ax=next(axis))
    metrics.PrecisionRecallDisplay.from_estimator(model, val_df, val_target, ax=next(axis))
    metrics.PredictionErrorDisplay.from_estimator(model, val_df, val_target, ax=next(axis))
    metrics.RocCurveDisplay.from_estimator(model, val_df, val_target, ax=next(axis))
    calibration.CalibrationDisplay.from_estimator(model, val_df, val_target, ax=next(axis))
    model_selection.LearningCurveDisplay.from_estimator(
        model, train_df, train_target, ax=next(axis)
    )
    return scores


def finalize_and_predict(model):
    model.fit(kaggle_train_df.drop(columns="Survived"), kaggle_train_df.Survived)
    return predict(model, kaggle_test_df)


# -

final_model = build_model(classifier=XGBClassifier(**study.best_params))
val_score(final_model)

submit_df = finalize_and_predict(final_model)
submit_df.to_csv("submission.csv")
