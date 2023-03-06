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

# # Titanic Data explore

# %load_ext autoreload

# %autoreload 1
# %aimport titanic

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns
from IPython.display import display

from titanic import add_features, compare_col, create_axs, expand_cabin

# -

input_dir = Path("/kaggle/input/titanic")
if not input_dir.exists():
    input_dir = Path("data")

kaggle_train_df = pd.read_csv(input_dir / "train.csv", index_col=0)
kaggle_test_df = pd.read_csv(input_dir / "test.csv", index_col=0)
kaggle_data = pd.concat(
    [kaggle_train_df, kaggle_test_df], keys=["train", "test"], names=["src"]
)

# ## raw data

kaggle_data.sample(20).sort_index()

f"train_size: {len(kaggle_train_df)}, test_size: {len(kaggle_test_df)}"

f"survived: {kaggle_train_df.Survived.mean():.1%}, not-survived: {1 - kaggle_train_df.Survived.mean():.1%}"

# ### histograms

axs = create_axs()
for col in kaggle_data:
    if pd.api.types.is_numeric_dtype(kaggle_data[col]):
        kaggle_data[col].groupby("src").plot.hist(
            density=True,
            ax=next(axs),
            bins=range(10)
            if col in ["Parch", "SibSp"]
            else range(0, 100, 5)
            if col in ["Age"]
            else 10,
            title=col,
            legend=True,
            alpha=0.5,
        )
    else:
        col_df = kaggle_data.groupby(["src", col]).size().unstack(0, fill_value=0)
        if len(col_df) > 10:
            print(
                {
                    "col": col,
                    "n_unique": len(col_df),
                }
            )
            print(
                col_df.sum(1)
                .sort_values(ascending=False)
                .iloc[:5]
                .astype(int)
                .to_dict()
            )
            print()
        else:
            ax = next(axs)
            (col_df / kaggle_data.groupby("src").size()).plot.bar(title=col, ax=ax)

# ### missing values

(
    kaggle_test_df.dtypes.to_frame("dtype").join(
        pd.concat(
            [
                kaggle_data.isna().sum().rename("all n_na"),
                kaggle_train_df.isna().sum().rename("train n_na"),
                kaggle_test_df.isna().sum().rename("test n_na"),
                kaggle_data.nunique().rename("all n_unique"),
                kaggle_train_df.nunique().rename("train n_unique"),
                kaggle_test_df.nunique().rename("test n_unique"),
            ],
            axis=1,
        )
        .drop(index="Survived")
        .astype(int)
    )
)

msno.matrix(kaggle_data, figsize=(25, 15))


# ## add features


cabin_df = expand_cabin(kaggle_train_df.Cabin)
{col: cabin_df[col].unique() for col in cabin_df}

# +
kaggle_xdata = pd.concat(
    [kaggle_train_df.pipe(add_features), kaggle_test_df.pipe(add_features)],
    keys=["train", "test"],
    names=["src"],
)

kaggle_xdata
# -

# ### histogram

# +
axs = create_axs()

for col in kaggle_xdata:
    if pd.api.types.is_numeric_dtype(kaggle_xdata[col]):
        kaggle_xdata[col].astype(float).groupby("src").plot.hist(
            density=True,
            ax=next(axs),
            bins=range(0, 150, 10) if col in ["C_num"] else 10,
            title=col,
            legend=True,
            alpha=0.5,
        )
    else:
        col_df = kaggle_xdata.groupby(["src", col]).size().unstack(0, fill_value=0)
        if len(col_df) > 20:
            display(
                col,
                col_df.sum(1)
                .sort_values(ascending=False)
                .iloc[:5]
                .astype(int)
                .to_dict(),
            )
        else:
            ax = next(axs)
            (col_df / kaggle_xdata.groupby("src").size()).plot.bar(title=col, ax=ax)

ax = next(axs)
sns.heatmap(
    kaggle_train_df.groupby("Cabin")
    .Survived.agg(["count", "sum"])
    .rename(columns={"count": "cabin size", "sum": "survived"})
    .value_counts()
    .sort_index()
    .unstack(),
    annot=True,
    ax=ax,
)
ax.set_title("C_count")
# -

kaggle_xdata.isna().sum()

# ## compare survival

fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(kaggle_train_df, "Fare").plot.bar(ax=axs[0])
compare_col(kaggle_train_df, "Age").plot.bar(ax=axs[1])

# ## Family size and correlation inside families

kaggle_train_df.eval("family_size=Parch+SibSp").join(
    kaggle_train_df.Name.str.extract(r"(.*),.*")[0].rename("FamilyName")
).plot.scatter(x="FamilyName", y="family_size")

# +
family_groups = 0
family_size = (
    kaggle_train_df.eval("family_size=Parch+SibSp")
    .join(kaggle_train_df.Name.str.extract(r"(.*),.*")[0].rename("FamilyName"))
    .groupby("FamilyName")
    .family_size.agg(["count", "median"])
    .sort_values("count", ascending=False)
    .rename(columns={"count": "members", "median": "median family_size of members"})
    .query("members>2")
)
ax = family_size.plot.bar(rot=90)

# TODO: add scatterplot
# -

survived_families = (
    kaggle_train_df.groupby(
        kaggle_train_df.Name.str.extract(r"(.*),.*")[0].rename("FamilyName")
    )
    .Survived.agg(["count", "sum"])
    .rename(columns={"sum": "survived"})
    .eval("not_survived=count-survived")
    .query("count>1")
    .sort_values("count", ascending=False)
    .drop(columns=["count"])
)
survived_families.head(50).plot.bar()
