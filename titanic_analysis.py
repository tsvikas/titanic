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
# TODO: search hp (optuna?)
# TODO: try ensemble.AdaBoostClassifier
# TODO: validation graphs
# TODO: read about XGBoost
# TODO: features: cabin is even?
# TODO: encode only common things
# TODO: remove "is zero"
# -

# # titanic model

# %load_ext autoreload

# %autoreload 1
# %aimport titanic

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn import calibration, impute, metrics, model_selection, preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import ParameterGrid, cross_validate, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from xgboost import XGBClassifier

from titanic import DataFrameDisplay, add_features, compare_col, create_axs

# -


# ## load the data

input_dir = Path("/kaggle/input/titanic")
if not input_dir.exists():
    input_dir = Path("data")

kaggle_train_df = pd.read_csv(input_dir / "train.csv", index_col=0)
kaggle_test_df = pd.read_csv(input_dir / "test.csv", index_col=0)
kaggle_data = pd.concat([kaggle_train_df, kaggle_test_df], keys=["train", "test"], names=["src"])

kaggle_data.sample(5).sort_index()

f"train_size: {len(kaggle_train_df)}, test_size: {len(kaggle_test_df)}"

f"survived: {kaggle_train_df.Survived.mean():.1%}, not-survived: {1 - kaggle_train_df.Survived.mean():.1%}"

# ### add features

ticket_group_sizes = pd.read_csv("data/ticket_group_sizes.csv", index_col=0).squeeze("columns")
kaggle_xdata = add_features(kaggle_data, ticket_group_sizes=ticket_group_sizes)
kaggle_xdata.sample(5).sort_index()

# ### missing values

kaggle_data.join(kaggle_xdata).pipe(
    lambda df: df.isna().sum().to_frame("n_na").join(df.nunique().rename("n_unique"))
)


# ## sklearn

# ### split

# +
# add validation for final estimation of the model
train_df, val_df = train_test_split(
    kaggle_train_df, test_size=100, stratify=kaggle_train_df.Survived, random_state=0
)
train_target = train_df.pop("Survived")
val_target = val_df.pop("Survived")

len(train_df), len(val_df)
# -

# ### preprocess


class AddColumns(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, func, kw_args):
        self.func = func
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        X_transformed = self._transform(X)
        self.feature_names_out_ = list(X_transformed.columns)
        return self

    def _transform(self, X):
        return X.join(self.func(X, **self.kw_args))

    def transform(self, X):
        X_transformed = self._transform(X)
        assert list(X_transformed.columns) == self.feature_names_out_
        return X_transformed

    def get_feature_names_out(self, feature_names_in=None):
        sklearn.utils.validation.check_is_fitted(self)
        if feature_names_in is None:
            feature_names_out = self.feature_names_out_
        else:
            assert len(feature_names_in) == len(self.feature_names_in_)
            feature_names_out = (
                list(feature_names_in) + self.feature_names_out_[len(feature_names_in) :]
            )
        return np.array(feature_names_out, dtype=object)


# +
class RenameFeatures(preprocessing.FunctionTransformer):
    def __init__(self, feature_name_fn):
        self.feature_name_fn = feature_name_fn
        super().__init__(
            feature_names_out=lambda s, input_features: [
                s.feature_name_fn(name) for name in input_features
            ]
        )


class DenseTransformer(preprocessing.FunctionTransformer):
    def __init__(self):
        super().__init__(
            func=lambda data: data.todense().A if hasattr(data, "todense") else data,
            feature_names_out="one-to-one",
        )


# +
def build_preprocess(
    use_family_name=False,  # bring too much data
    # use_first_name=False,     # bring too much data
    use_cabin_prefix=True,  # have many na
    use_cabin_num=False,  # bring too much data + many na
    use_cabin_full=False,  # bring too much data + many na
    use_ticket_prefix=True,  # have unequal dist?
    use_ticket_number=False,  # have too much data?
    use_ticket_group_size=True,  # pseudo leakage
    scale_numerical_cols=False,  # not needed in trees
    bin_numerical_cols=False,  # not needed in trees
):
    scaling_cls = preprocessing.StandardScaler if scale_numerical_cols else lambda: "passthrough"

    # add features
    features_creator = AddColumns(add_features, kw_args=dict(ticket_group_sizes=ticket_group_sizes))

    # split and handle columns
    ## not imputing
    cabin_transformer = ColumnTransformer(
        [
            (
                "cabin_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_cabin_full
                else "drop",
                ["Cabin"],
            ),
            (
                "c_prefix_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_cabin_prefix
                else "drop",
                ["C_letter"],
            ),
            (
                "c_num_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_cabin_num
                else "drop",
                ["C_num"],
            ),
            (
                "c_count",
                make_pipeline(SimpleImputer(strategy="constant", fill_value=0), scaling_cls()),
                ["C_count"],
            ),
            ("cabin_missing", impute.MissingIndicator(features="all"), ["Cabin"]),
        ]
    )
    ## yes imputing
    count_names = Pipeline(
        [
            ("count", CountVectorizer()),
            ("improve_names", RenameFeatures(lambda name: "FirstName_" + name)),
        ]
    )
    pre_impute_transformer = ColumnTransformer(
        [
            (
                "encode",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Pclass", "Embarked", "PrefixName"],
            ),
            (
                "encode_binary",
                preprocessing.OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="if_binary"
                ),
                ["Sex", "is_zero_fare", "travel_alone"],
            ),
            (
                "ticket_prefix_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_ticket_prefix
                else "drop",
                ["ticket_prefix"],
            ),
            (
                "ticket_number_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_ticket_number
                else "drop",
                ["ticket_number"],
            ),
            (
                "ticket_group_size",
                scaling_cls() if use_ticket_group_size else "drop",
                ["ticket_group_size"],
            ),
            (
                "name_family_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_family_name
                else "drop",
                ["FamilyName"],
            ),
            # bug: no set_output()
            # ('name_first_vec',   count_names if use_first_name else 'drop', 'FirstName'),
            ("normalize", scaling_cls(), ["Age", "SibSp", "Parch", "Fare", "family_size"]),
            (
                "binning",
                make_pipeline(
                    SimpleImputer(strategy="mean"),
                    preprocessing.KBinsDiscretizer(encode="onehot-dense", random_state=0),
                )
                if bin_numerical_cols
                else "drop",
                ["Age", "Fare"],
            ),
            ("name_drop", "drop", ["Name"]),
            ("ticket_drop", "drop", ["Ticket"]),
            ("missing_ind", impute.MissingIndicator(features="all"), ["Age", "Embarked", "Fare"]),
        ]
    )

    # TODO: add to args?
    # TODO: needs scaling?
    imputer = KNNImputer()

    main_transformer = Pipeline(
        [
            ("pre_impute_transformer", pre_impute_transformer),
            ("to_dense", DenseTransformer()),
            ("imputer", imputer),
        ]
    )

    # compose together
    features_transformer = FeatureUnion(
        [("cabin_transformer", cabin_transformer), ("main_transformer", main_transformer)]
    )

    renamer = RenameFeatures(lambda name: name.rsplit("__")[-1].replace(".0", ""))
    data_transformer = Pipeline(
        [
            ("features_creator", features_creator),
            ("features_transformer", features_transformer),
            ("renamer", renamer),
            ("to_dense", DenseTransformer()),
        ]
    ).set_output(transform="pandas")

    return data_transformer


build_preprocess()
# -
# #### test preprocess

build_preprocess().fit(train_df).get_feature_names_out()

build_preprocess().fit_transform(train_df)


# ### model


# +
def build_model(transformer=None, classifier=None):
    if transformer is None:
        transformer = build_preprocess()
    if classifier is None:
        classifier = RandomForestClassifier()

    pipeline = Pipeline([("transformer", transformer), ("classifier", classifier)])
    return pipeline


train_df_shuffled = train_df.sample(frac=1, random_state=1)
train_target_shuffled = train_target.reindex_like(train_df_shuffled)
model = build_model()
model[:-1].fit_transform(train_df_shuffled, train_target_shuffled).sort_index()
# -

# #### test model

parameters = list(inspect.signature(build_preprocess).parameters)
# skipping: use_first_name
for param_name in parameters:
    print(param_name)
    kw_args = {p: p == param_name for p in parameters}
    model = build_model(build_preprocess(**kw_args))
    assert model.fit(train_df, train_target).score(train_df, train_target) > 0.5

# ### evaluate

metrics.SCORERS.keys()


# +
def format_scores(scores, add_low=False):
    m = scores.mean()
    s = scores.std()
    return f"{m:.2%} ±{s:.2%}" + (f" [{m-s:.2%}]" if add_low else "")


def evaluate_model(model, verbose=False, enhance_scores=False, cv=10, random_shuffle=False):
    train_df_shuffled = train_df
    if random_shuffle is not False:
        if random_shuffle is True:
            random_shuffle = None
        train_df_shuffled = train_df_shuffled.sample(frac=1, random_state=random_shuffle)
    train_target_shuffled = train_target.reindex_like(train_df_shuffled)
    scores = cross_validate(
        model,
        train_df_shuffled,
        train_target_shuffled,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "roc_auc", "f1"],
        n_jobs=-1,
    )
    if verbose:
        for name, score in scores.items():
            if "time" in name:
                print(f"{name:24}: {score.mean():5.2} sec")
            else:
                print(f"{name:24}: " + format_scores(score))
    if enhance_scores:
        new_scores = {}
        for name, score in scores.items():
            if "time" in name:
                new_scores[name] = score.mean()
            else:
                new_scores[name.replace("test_", "") + "_μ"] = score.mean()
                new_scores[name.replace("test_", "") + "_σ"] = score.std()
        return new_scores
    return scores


# -

# #### test evaluate and repeatability

# +
model = build_model(
    build_preprocess(
        use_family_name=True,  # bring too much data
        # use_first_name=True,        # bring too much data
        use_cabin_prefix=True,  # have many na
        use_cabin_num=True,  # bring too much data + many na
        use_cabin_full=True,  # bring too much data + many na
        use_ticket_prefix=True,  # have many na?
        use_ticket_number=True,
        use_ticket_group_size=True,
        scale_numerical_cols=False,  # not needed in trees
    ),
    classifier=RandomForestClassifier(random_state=0),
)

results = DataFrameDisplay()
for i in range(2):
    display("Running...")
    results.add_row(evaluate_model(model, enhance_scores=True, random_shuffle=False), f"d_{i}")

for i in range(3):
    display("Running...")
    results.add_row(
        evaluate_model(
            build_model(classifier=RandomForestClassifier(random_state=0)),
            enhance_scores=True,
            random_shuffle=True,
        ),
        f"shuffle_{i}",
    )

for i in range(3):
    display("Running...")
    results.add_row(
        evaluate_model(
            build_model(classifier=RandomForestClassifier(random_state=None)),
            enhance_scores=True,
            random_shuffle=False,
        ),
        f"classifier_{i}",
    )
# -


# ### find best model

# #### which features?

# +
fn_1 = Path("cache/1.csv")
params_1 = [
    "use_family_name",
    "use_cabin_prefix",
    "use_cabin_num",
    "use_cabin_full",
    "use_ticket_prefix",
]
param_grid_1 = ParameterGrid({p: [False, True] for p in params_1})

if fn_1.exists():
    results_1 = DataFrameDisplay.load(fn_1, params_1)
    display(results_1.df)
else:
    results_1 = DataFrameDisplay(params_1)
    for param in param_grid_1:
        model = build_model(
            transformer=build_preprocess(**param, scale_numerical_cols=False),
            classifier=XGBClassifier(),
        )
        display("Running...")
        result = evaluate_model(model, enhance_scores=True)
        result["num_features"] = len(model[:-1].fit(train_df).get_feature_names_out())
        results_1.add_row(result, param)
    results_1.save(fn_1)
# -

results_1.df.plot.scatter(x="num_features", y="accuracy_μ", yerr="accuracy_σ")

results_1.df.plot.scatter(x="num_features", y="fit_time")

# #### which hyperparams?

# ##### 2

# +
# %%time

param_grid_2 = {
    "classifier__n_estimators": [2, 10, 20, 50, 150, 250],
    "classifier__max_depth": [3, 8],
    "classifier__learning_rate": [1.0, 0.1, 0.02, 0.01, 0.005],
}
search_2 = model_selection.GridSearchCV(
    build_model(
        transformer=build_preprocess(), classifier=XGBClassifier(objective="binary:logistic")
    ),
    param_grid_2,
    scoring="accuracy",
    n_jobs=-1,
).fit(train_df, train_target)
results_2 = pd.DataFrame(search_2.cv_results_)
# -

results_2.plot.scatter(
    x="mean_fit_time", y="mean_test_score", yerr="std_test_score", alpha=0.5, logx=True
)

grid = (
    results_2.rename(columns=lambda s: s.replace("param_classifier__", ""))
    .set_index(["learning_rate", "max_depth", "n_estimators"])
    .mean_test_score.to_xarray()
    .plot(col="max_depth", xscale="log", yscale="log")
)
for ax in grid.axs.reshape(-1):
    ax.plot(50, 0.01, "rx")

# let's look around \[50, *, 0.01]

# ##### 3

# +
# %%time

param_grid_3 = {
    "classifier__n_estimators": [25, 50, 100],
    "classifier__max_depth": [3, 5, 7, 10],
    "classifier__learning_rate": [0.0025, 0.005, 0.01, 0.025, 0.05],
}
search_3 = model_selection.GridSearchCV(
    build_model(
        transformer=build_preprocess(), classifier=XGBClassifier(objective="binary:logistic")
    ),
    param_grid_3,
    scoring="accuracy",
    n_jobs=-1,
).fit(train_df, train_target)
results_3 = pd.DataFrame(search_3.cv_results_)
# -

results_3.plot.scatter(
    x="mean_fit_time", y="mean_test_score", yerr="std_test_score", alpha=0.5, logx=True
)

grid = (
    results_3.rename(columns=lambda s: s.replace("param_classifier__", ""))
    .set_index(["learning_rate", "max_depth", "n_estimators"])
    .mean_test_score.to_xarray()
    .plot(col="max_depth", xscale="log", yscale="log")
)
for ax in grid.axs.reshape(-1):
    ax.plot(100, 0.05, "rx")

# let's look around \[100, 1-6, 0.05\]

# ##### 4

# +
# %%time

param_grid_4 = {
    "classifier__n_estimators": [75, 100, 125, 150, 200, 300],
    "classifier__max_depth": [1, 2, 3, 4, 5, 6],
    "classifier__learning_rate": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
}
search_4 = model_selection.GridSearchCV(
    build_model(
        transformer=build_preprocess(), classifier=XGBClassifier(objective="binary:logistic")
    ),
    param_grid_4,
    scoring="accuracy",
    n_jobs=-1,
).fit(train_df, train_target)
results_4 = pd.DataFrame(search_4.cv_results_)
# -

results_4.plot.scatter(
    x="mean_fit_time", y="mean_test_score", yerr="std_test_score", alpha=0.5, logx=True
)

grid = (
    results_4.rename(columns=lambda s: s.replace("param_classifier__", ""))
    .set_index(["learning_rate", "max_depth", "n_estimators"])
    .mean_test_score.to_xarray()
    .plot(col="max_depth", xscale="log", yscale="log")
)
for ax in grid.axs.reshape(-1):
    ax.plot(150, 0.08, "rx")

# #### look at all runs

all_results = (
    pd.concat([results_2, results_3, results_4])
    .rename(columns=lambda s: s.replace("param_classifier__", ""))
    .set_index(["learning_rate", "max_depth", "n_estimators"])
)

all_results[["mean_test_score", "std_test_score"]].sort_values("mean_test_score", ascending=False)

# ##### times

# +
time_df = all_results.mean_fit_time.reset_index()
axs = create_axs(3)
time_df.plot.scatter(x="n_estimators", y="mean_fit_time", ax=next(axs), s=1)
time_df.plot.scatter(x="max_depth", y="mean_fit_time", ax=next(axs), s=1)
time_df.plot.scatter(x="learning_rate", y="mean_fit_time", ax=next(axs), s=1)

time_df.query("n_estimators == 300").plot.scatter(
    x="max_depth", y="mean_fit_time", ax=next(axs), s=1
)
time_df.query("n_estimators == 300").plot.scatter(
    x="learning_rate", y="mean_fit_time", ax=next(axs), s=1
)
time_df.groupby(["n_estimators", "max_depth"]).mean_fit_time.median().to_xarray().plot(ax=next(axs))
# -

# ### understand

# +
model = build_model(classifier=XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01))
model.fit(train_df, train_target)

features_df = pd.Series(
    model[-1].feature_importances_, index=model[-1].feature_names_in_
).sort_values(ascending=False)
features_df.head(10).plot.bar()
# -

fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(train_df, train_target, "PrefixName_Mr", model).plot.bar(
    ax=axs[0], ylim=(0, len(train_df))
)
compare_col(train_df, train_target, "PrefixName_Miss", model).plot.bar(
    ax=axs[1], ylim=(0, len(train_df))
)

fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(train_df, train_target, "Fare").plot.bar(ax=axs[0])
compare_col(train_df, train_target, "Age").plot.bar(ax=axs[1])


# ### finalize_and_predict


# +
def predict(model, data):
    return pd.Series(model.predict(data), index=data.index, name="Survived")


def val_score(model):
    model.fit(train_df, train_target)
    scores = {}
    scores["score"] = model.score(val_df, val_target)
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

    axs = create_axs(2, ax_size=(6, 4))
    metrics.ConfusionMatrixDisplay.from_estimator(
        model, val_df, val_target, display_labels=["ns", "s"], ax=next(axs)
    )
    metrics.DetCurveDisplay.from_estimator(model, val_df, val_target, ax=next(axs))
    metrics.PrecisionRecallDisplay.from_estimator(model, val_df, val_target, ax=next(axs))
    metrics.PredictionErrorDisplay.from_estimator(model, val_df, val_target, ax=next(axs))
    metrics.RocCurveDisplay.from_estimator(model, val_df, val_target, ax=next(axs))
    calibration.CalibrationDisplay.from_estimator(model, val_df, val_target, ax=next(axs))
    model_selection.LearningCurveDisplay.from_estimator(model, train_df, train_target, ax=next(axs))
    return scores


def finalize_and_predict(model):
    model.fit(kaggle_train_df.drop(columns="Survived"), kaggle_train_df.Survived)
    return predict(model, kaggle_test_df)


# -

# final_model = build_model(
#     True,
#     KNNImputer,
#     {},
#     XGBClassifier,
#     {"n_estimators": 20, "max_depth": 8, "learning_rate": 0.1, "objective": "binary:logistic"},
# )
# final_model = build_model(True, KNNImputer, {}, RandomForestClassifier, {'n_estimators': 100})
final_model = build_model(
    classifier=XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.08)
)
val_score(final_model)

submit_df = finalize_and_predict(final_model)
submit_df.to_csv("submission.csv")
