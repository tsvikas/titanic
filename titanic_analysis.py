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

# # titanic model

# %load_ext autoreload

# %autoreload 1
# %aimport titanic

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# +
from sklearn import impute, metrics, preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import cross_validate
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

kaggle_xdata = add_features(kaggle_data)
kaggle_xdata.sample(5).sort_index()

# ### missing values

kaggle_data.join(kaggle_xdata).pipe(
    lambda df: df.isna().sum().to_frame("n_na").join(df.nunique().rename("n_unique"))
)


# ## sklearn

# ### split

# +
# train_df, val_df = train_test_split(kaggle_train_df, stratify=kaggle_train_df.Survived)
# train_target = train_df.pop("Survived")
# val_target = val_df.pop("Survived")

# len(train_df), len(val_df)
# -

# ### preprocess


# +
class AddColumns(preprocessing.FunctionTransformer):
    def __init__(self, get_features):
        self.get_features = get_features
        super().__init__(
            func=lambda df: df.join(get_features(df)),
            feature_names_out=lambda s, input_features: list(input_features) + s.get_features.names,
        )


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
# TODO: better imputer!
# TODO: search hp (optuna?)
# TODO: use black -l 120
# TODO: try ensemble.AdaBoostClassifier
def build_preprocess(
    use_family_name=False,  # bring too much data
    # use_first_name=False,     # bring too much data
    use_cabin_prefix=True,  # have many na
    use_cabin_num=False,  # bring too much data + many na
    use_cabin_full=False,  # bring too much data + many na
    use_ticket_prefix=True,  # have many na?
    scale_numerical_cols=False,  # not needed in trees
):
    scaling_cls = preprocessing.StandardScaler if scale_numerical_cols else lambda: "passthrough"

    # add features
    features_creator = AddColumns(add_features)

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
                ["Sex", "is_zero_fare"],
            ),
            (
                "t1_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_ticket_prefix
                else "drop",
                ["ticket_prefix"],
            ),
            (
                "name_family_enc",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_family_name
                else "drop",
                ["FamilyName"],
            ),
            # ('name_first_vec',   count_names if use_first_name else 'drop', 'FirstName'),
            ("normalize", scaling_cls(), ["Age", "SibSp", "Parch", "Fare"]),
            ("name_drop", "drop", ["Name"]),
            ("ticket_drop", "drop", ["Ticket"]),
            ("missing_ind", impute.MissingIndicator(features="all"), ["Age", "Embarked", "Fare"]),
            # TODO: add these
            #             ('age_bin',          preprocessing.KBinsDiscretizer(), ['Age']),
            #             ('fare_bin',         preprocessing.KBinsDiscretizer(), ['Fare']),
        ]
    )

    # note - imputer can't work with spars
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

    renamer = RenameFeatures(lambda name: name.rsplit("__")[-1])
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

build_preprocess().fit(kaggle_train_df).get_feature_names_out()

build_preprocess().fit_transform(kaggle_train_df)


# ### model


# +
def build_model(transformer=None, classifier=None):
    if transformer is None:
        transformer = build_preprocess()
    if classifier is None:
        classifier = RandomForestClassifier()

    pipeline = Pipeline([("transformer", transformer), ("classifier", classifier)])
    return pipeline


kaggle_train_df_shuffled = kaggle_train_df.sample(frac=1, random_state=1)
model = build_model()
model[:-1].fit_transform(kaggle_train_df_shuffled, kaggle_train_df_shuffled.Survived).sort_index()
# -

# #### test model

props = "use_family_name use_cabin_prefix use_cabin_num use_cabin_full use_ticket_prefix scale_numerical_cols".split()
# skipping: use_first_name
for prop in props:
    print(prop)
    kw_args = {p: p == prop for p in props}
    model = build_model(build_preprocess(**kw_args))
    model[:-1].fit_transform(kaggle_train_df, kaggle_train_df.Survived)

# ### evaluate

metrics.SCORERS.keys()


# +
# to be used with train_test_split
# def evaluate_classifier(model, verbose=True):
#     model.fit(train_df, y=train_target)
#     # train_pred = pd.Series(model.predict(train_df), index=train_df.index, name="Survived_pred")
#     val_pred = pd.Series(model.predict(val_df), index=val_df.index, name="Survived_pred")
#     if verbose:
#         print(metrics.classification_report(val_target, val_pred, target_names=['ns', 's']))
#         print("accuracy_score =", metrics.accuracy_score(val_target, val_pred))
#         print("balanced_accuracy_score =", metrics.balanced_accuracy_score(val_target, val_pred))
#         val_prob = model.predict_proba(val_df)[:, 1]
#         print("roc_auc_score =", metrics.roc_auc_score(val_target, val_prob))
#     return model, val_pred


def format_scores(scores, add_low=False):
    m = scores.mean()
    s = scores.std()
    return f"{m:.2%} ±{s:.2%}" + (f" [{m-s:.2%}]" if add_low else "")


def evaluate_model(model, verbose=False, enhance_scores=False, cv=10, random_shuffle=False):
    kaggle_train_df_shuffled = kaggle_train_df
    if random_shuffle is not False:
        if random_shuffle is True:
            random_shuffle = None
        kaggle_train_df_shuffled = kaggle_train_df_shuffled.sample(
            frac=1, random_state=random_shuffle
        )
    scores = cross_validate(
        model,
        kaggle_train_df_shuffled,
        kaggle_train_df_shuffled.Survived,
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


# +
# to be used with train_test_split
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# metrics.ConfusionMatrixDisplay.from_predictions(
#     val_target, val_pred, display_labels=["ns", "s"], ax=axs[0]
# )
# metrics.RocCurveDisplay.from_estimator(model, val_df, val_target, ax=axs[1]);
# -

# ### find best model

# #### which features?

results_1 = DataFrameDisplay(
    "family_name,cabin_prefix,cabin_num,cabin_full,ticket_prefix".split(",")
)
for (
    use_family_name,
    use_cabin_prefix,
    use_cabin_num,
    use_cabin_full,
    use_ticket_prefix,
) in itertools.product([False, True], repeat=5):
    kwargs: dict[str, Any] = dict(
        use_family_name=use_family_name,
        # use_first_name=False,
        use_cabin_prefix=use_cabin_prefix,
        use_cabin_num=use_cabin_num,
        use_cabin_full=use_cabin_full,
        use_ticket_prefix=use_ticket_prefix,
    )
    model = build_model(
        transformer=build_preprocess(**kwargs, scale_numerical_cols=False),
        classifier=XGBClassifier(),
    )
    display("Running...")
    result = evaluate_model(model, enhance_scores=True)
    result["num_features"] = len(model[:-1].fit(kaggle_train_df).get_feature_names_out())
    results_1.add_row(result, list(kwargs.values()))

results_1.df.plot.scatter(x="num_features", y="accuracy_μ", yerr="accuracy_σ")

results_1.df.plot.scatter(x="num_features", y="fit_time")

# #### which hyperparams?

# ##### 2

# +
results_2 = DataFrameDisplay("n_estimators max_depth learning_rate".split())

for n_estimators, max_depth, learning_rate in itertools.product(
    [2, 10, 20, 50, 150, 250], [3, 8], [1.0, 0.1, 0.02, 0.01, 0.005]
):
    kwargs = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    model = build_model(
        transformer=build_preprocess(),
        classifier=XGBClassifier(**kwargs, objective="binary:logistic"),
    )
    display("Running...")
    result = evaluate_model(model, enhance_scores=True)
    results_2.add_row(result, list(kwargs.values()))

# +
# results_2.df.plot.line(y='accuracy_μ', yerr='accuracy_σ', marker='.', linestyle='', rot=90)
# -

results_2.df.plot.scatter(x="fit_time", y="accuracy_μ", yerr="accuracy_σ", alpha=0.5, logx=True)

results_2.df.accuracy_μ.to_xarray().plot(col="learning_rate", cmap="gray")

# ##### 3

# +
results_3 = DataFrameDisplay("n_estimators max_depth learning_rate".split())

for n_estimators, max_depth, learning_rate in itertools.product(
    [25, 50, 100], [3, 5, 7, 10], [0.0025, 0.005, 0.01, 0.025, 0.05]
):
    kwargs_3: dict[str, Any] = dict(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
    )
    model = build_model(
        transformer=build_preprocess(),
        classifier=XGBClassifier(**kwargs_3, objective="binary:logistic"),
    )
    display("Running...")
    result = evaluate_model(model, enhance_scores=True)
    results_3.add_row(result, list(kwargs_3.values()))
# -

results_3.df.plot.scatter(x="fit_time", y="accuracy_μ", yerr="accuracy_σ", alpha=0.5, logx=True)

results_3.df.accuracy_μ.to_xarray().plot(col="learning_rate", cmap="gray")

# ##### 4

# +
results_4 = DataFrameDisplay("n_estimators max_depth learning_rate".split())

for n_estimators, max_depth, learning_rate in itertools.product(
    [75, 100, 125, 150, 200, 300],
    [1, 2, 3, 4, 5, 6],
    [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
):
    kwargs_4 = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    model = build_model(
        transformer=build_preprocess(),
        classifier=XGBClassifier(**kwargs_4, objective="binary:logistic"),
    )
    display("Running...")
    result = evaluate_model(model, enhance_scores=True)
    results_4.add_row(result, list(kwargs_4.values()))
# -

results_4.df.plot.scatter(x="fit_time", y="accuracy_μ", yerr="accuracy_σ", alpha=0.5, logx=True)

results_4.df.accuracy_μ.to_xarray().plot(col="learning_rate", cmap="gray")

# #### look at all runs

all_results = pd.concat([results_2.df, results_3.df, results_4.df])

all_results.sort_values("accuracy_μ", ascending=False)

# ##### best

results_best = DataFrameDisplay()
for i in range(10):
    display("Running...")
    results_best.add_row(
        evaluate_model(
            build_model(
                classifier=XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.08)
            ),
            enhance_scores=True,
            random_shuffle=True,
        ),
        i,
    )


results_best_2 = DataFrameDisplay()
for i in range(10):
    display("Running...")
    results_best_2.add_row(
        evaluate_model(
            build_model(
                transformer=build_preprocess(use_family_name=True, use_cabin_full=True),
                classifier=XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.08),
            ),
            enhance_scores=True,
            random_shuffle=True,
        ),
        i,
    )


ax = results_best.df.plot(y="accuracy_μ", yerr="accuracy_σ", marker=".", linestyle="")
ax = results_best_2.df.plot(
    y="accuracy_μ", yerr="accuracy_σ", marker=".", linestyle="", ax=ax, alpha=0.5
)

# ##### times

# +
time_df = all_results.fit_time.reset_index()
axs = create_axs(3)
time_df.plot.scatter(x="n_estimators", y="fit_time", ax=next(axs), s=1)
time_df.plot.scatter(x="max_depth", y="fit_time", ax=next(axs), s=1)
time_df.plot.scatter(x="learning_rate", y="fit_time", ax=next(axs), s=1)

time_df.query("n_estimators == 300").plot.scatter(x="max_depth", y="fit_time", ax=next(axs), s=1)
time_df.query("n_estimators == 300").plot.scatter(
    x="learning_rate", y="fit_time", ax=next(axs), s=1
)
time_df.groupby(["n_estimators", "max_depth"]).fit_time.median().to_xarray().plot(ax=next(axs))
# -

# ### understand

# +
model = build_model(classifier=XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01))
model.fit(kaggle_train_df, kaggle_train_df.Survived)

features_df = pd.Series(
    model[-1].feature_importances_, index=model[-1].feature_names_in_
).sort_values(ascending=False)
features_df.head(10).plot.bar()
# -

fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(kaggle_train_df, "PrefixName_Mr", model).plot.bar(
    ax=axs[0], ylim=(0, len(kaggle_train_df))
)
compare_col(kaggle_train_df, "PrefixName_Miss", model).plot.bar(
    ax=axs[1], ylim=(0, len(kaggle_train_df))
)

fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(kaggle_train_df, "Fare").plot.bar(ax=axs[0])
compare_col(kaggle_train_df, "Age").plot.bar(ax=axs[1])


# ### finalize_and_predict


# +
def finalize_and_predict(model):
    model.fit(kaggle_train_df.drop(columns="Survived"), kaggle_train_df.Survived)
    return pd.Series(
        data=model.predict(kaggle_test_df), index=kaggle_test_df.index, name="Survived"
    )


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
submit_df = finalize_and_predict(final_model)
submit_df.to_csv("submission.csv")
submit_df
