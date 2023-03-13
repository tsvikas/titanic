# # titanic model

# %load_ext autoreload

# %autoreload 1
# %aimport titanic

# +
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn import impute, preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

from titanic.features import add_features
from titanic.notebook import DataFrameDisplay

try:
    data_dir = Path(__file__).parents[1] / "data"
except NameError:
    data_dir = Path("../data")
ticket_group_sizes = pd.read_csv(data_dir / "ticket_group_sizes.csv", index_col=0).squeeze(
    "columns"
)
# -
# ## load the data

if __name__ == "__main__":
    kaggle_train_df = pd.read_csv(data_dir / "train.csv", index_col=0)
    kaggle_test_df = pd.read_csv(data_dir / "test.csv", index_col=0)
    train_df = kaggle_train_df.copy()
    train_target = train_df.pop("Survived")

    kaggle_data = pd.concat(
        [kaggle_train_df, kaggle_test_df], keys=["train", "test"], names=["src"]
    )
    display(kaggle_data.sample(5).sort_index())
    display(kaggle_data.groupby("src").size().to_frame("count"))
    display(
        kaggle_train_df.groupby("Survived")
        .size()
        .to_frame("count")
        .assign(percent=lambda df: df / df.sum())
    )

if __name__ == "__main__":
    kaggle_xdata = add_features(kaggle_data, ticket_group_sizes=ticket_group_sizes)
    display(kaggle_xdata.dropna().sample(5).sort_index())

# ### missing values

if __name__ == "__main__":
    display(
        kaggle_data.join(kaggle_xdata).pipe(
            lambda df: df.isna().sum().to_frame("n_na").join(df.nunique().rename("n_unique"))
        )
    )


# ## preprocess


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
                "c_num_even",
                preprocessing.FunctionTransformer(
                    lambda sr: (1 - (sr % 2)).fillna(0),
                    feature_names_out=lambda s, feature_names_in: [
                        n + "_even" for n in feature_names_in
                    ],
                ),
                ["C_num"],
            ),
            (
                "c_num_odd",
                preprocessing.FunctionTransformer(
                    lambda sr: (sr % 2).fillna(0),
                    feature_names_out=lambda s, feature_names_in: [
                        n + "_odd" for n in feature_names_in
                    ],
                ),
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
            (
                "normalize",
                scaling_cls(),
                ["Age", "SibSp", "Parch", "Fare", "family_size"],
            ),
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
            (
                "missing_ind",
                impute.MissingIndicator(features="all"),
                ["Age", "Embarked", "Fare"],
            ),
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
        [
            ("cabin_transformer", cabin_transformer),
            ("main_transformer", main_transformer),
        ]
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


# ### test preprocess

if __name__ == "__main__":
    build_preprocess()
    print("✓", "build")
    build_preprocess().fit(kaggle_train_df)
    print("✓", "fit")
    build_preprocess().fit_transform(kaggle_train_df)
    print("✓", "transform")
    build_preprocess().fit(kaggle_train_df).get_feature_names_out()
    print("✓", "feature_names")


# ## model


def build_model(transformer=None, classifier=None):
    if transformer is None:
        transformer = build_preprocess()
    if classifier is None:
        classifier = RandomForestClassifier()

    pipeline = Pipeline([("transformer", transformer), ("classifier", classifier)])
    return pipeline


# ### test model

if __name__ == "__main__":
    model = build_model()
    print("✓", "build")
    model[:-1].fit_transform(train_df, train_target)
    print("✓", "preprocess fit_transform")

    parameters = list(inspect.signature(build_preprocess).parameters)
    # skipping: use_first_name
    for param_name in parameters:
        kw_args = {p: p == param_name for p in parameters}
        model = build_model(build_preprocess(**kw_args))
        assert model.fit(train_df, train_target).score(train_df, train_target) > 0.5
        print("✓", param_name)

# ## evaluate


# +
def format_scores(scores, add_low=False):
    m = scores.mean()
    s = scores.std()
    return f"{m:.2%} ±{s:.2%}" + (f" [{m-s:.2%}]" if add_low else "")


def evaluate_model(model, X, y, cv=10, random_shuffle=False):
    if random_shuffle is not False:
        if random_shuffle is True:
            random_shuffle = None
        X = X.sample(frac=1, random_state=random_shuffle)
        y = y.reindex_like(X)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "roc_auc", "f1"],
        n_jobs=-1,
    )
    new_scores = {}
    for name, score in scores.items():
        if "time" in name:
            new_scores[name] = score.mean()
        else:
            new_scores[name.replace("test_", "") + "_μ"] = score.mean()
            new_scores[name.replace("test_", "") + "_σ"] = score.std()
    return new_scores


# -

# ### test evaluate and repeatability

if __name__ == "__main__":
    all_preprocess_options = dict(
        use_family_name=True,  # bring too much data
        # use_first_name=True,        # bring too much data
        use_cabin_prefix=True,  # have many na
        use_cabin_num=True,  # bring too much data + many na
        use_cabin_full=True,  # bring too much data + many na
        use_ticket_prefix=True,  # have many na?
        use_ticket_number=True,
        use_ticket_group_size=True,
        scale_numerical_cols=False,  # not needed in trees
    )

    results = DataFrameDisplay()
    for i in range(2):
        display("Running...")
        results.add_row(
            evaluate_model(
                build_model(
                    build_preprocess(**all_preprocess_options),
                    classifier=RandomForestClassifier(random_state=0),
                ),
                train_df,
                train_target,
                random_shuffle=False,
            ),
            f"d_{i}",
        )

    for i in range(3):
        display("Running...")
        results.add_row(
            evaluate_model(
                build_model(classifier=RandomForestClassifier(random_state=0)),
                train_df,
                train_target,
                random_shuffle=True,
            ),
            f"shuffle_{i}",
        )

    for i in range(3):
        display("Running...")
        results.add_row(
            evaluate_model(
                build_model(classifier=RandomForestClassifier(random_state=None)),
                train_df,
                train_target,
                random_shuffle=False,
            ),
            f"classifier_{i}",
        )
