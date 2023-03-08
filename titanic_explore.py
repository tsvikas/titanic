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

from titanic import add_features, compare_col, create_axs, expand_cabin

# -

input_dir = Path("/kaggle/input/titanic")
if not input_dir.exists():
    input_dir = Path("data")

kaggle_train_df = pd.read_csv(input_dir / "train.csv", index_col=0)
kaggle_test_df = pd.read_csv(input_dir / "test.csv", index_col=0)
kaggle_data = pd.concat([kaggle_train_df, kaggle_test_df], keys=["train", "test"], names=["src"])

# ## raw data

kaggle_data.sample(20).sort_index()

f"train_size: {len(kaggle_train_df)}, test_size: {len(kaggle_test_df)}"

f"survived: {kaggle_train_df.Survived.mean():.1%}, not-survived: {1 - kaggle_train_df.Survived.mean():.1%}"


# ### histograms


# +
def plot_histograms(df, split_col, bins, dropna=False):
    N_CATEGORICAL = 20
    axs = create_axs()
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > N_CATEGORICAL:
            ax = next(axs)
            df.groupby([split_col], dropna=dropna)[col].plot.hist(
                density=True,
                ax=ax,
                bins=bins.get(col, 10),
                title=col,
                legend=True,
                # alpha=0.5,
                histtype="step",
            )
            ax.legend(title=split_col)
        else:
            col_df = df.groupby([split_col, col], dropna=dropna).size().unstack(0, fill_value=0)
            if len(col_df) > N_CATEGORICAL:
                print({"col": col, "n_unique": len(col_df)})
                print(col_df.sum(1).sort_values(ascending=False).iloc[:5].astype(int).to_dict())
                print()
            else:
                ax = next(axs)
                (col_df / col_df.sum()).plot.bar(title=col, ax=ax)
    return axs


plot_histograms(kaggle_data, "src", {"Age": range(0, 100, 5), "Fare": range(0, 400, 10)})
# -

plot_histograms(kaggle_train_df, "Survived", {"Age": range(0, 100, 5), "Fare": range(0, 400, 10)})

# +
# for sex in kaggle_train_df["Sex"].unique():
#     plot_histograms(
#         kaggle_train_df.query("Sex==@sex"),
#         "Survived",
#         {"Age": range(0, 100, 5), "Fare": range(0, 400, 10)},
#     )
# -

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
ticket_group_sizes = pd.read_csv("data/ticket_group_sizes.csv", index_col=0).squeeze("columns")
kaggle_xdata = pd.concat(
    [
        kaggle_train_df.pipe(add_features, ticket_group_sizes=ticket_group_sizes),
        kaggle_test_df.pipe(add_features),
    ],
    keys=["train", "test"],
    names=["src"],
)

kaggle_xdata
# -

# ### histogram

axs = plot_histograms(kaggle_xdata, "src", {"C_num": range(0, 150, 10)}, dropna=True)

# +
axs = plot_histograms(
    kaggle_xdata.query("src=='train'").join(kaggle_train_df.Survived),
    "Survived",
    {"C_num": range(0, 150, 10)},
    dropna=True,
)

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

(
    kaggle_xdata.dtypes.to_frame("dtype").join(
        pd.concat(
            [
                kaggle_xdata.isna().sum().rename("all n_na"),
                kaggle_xdata.nunique().rename("all n_unique"),
            ],
            axis=1,
        ).astype(int)
    )
)

# ## compare survival

fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
compare_col(kaggle_train_df, kaggle_train_df.Survived, "Fare").plot.bar(ax=axs[0])
compare_col(kaggle_train_df, kaggle_train_df.Survived, "Age").plot.bar(ax=axs[1])

# ## Family size and correlation inside families

kaggle_train_df.join(kaggle_xdata).plot.scatter(x="FamilyName", y="family_size")

# +
family_groups = 0
family_size = (
    kaggle_xdata.groupby("FamilyName")
    .family_size.agg(["count", "median"])
    .sort_values("count", ascending=False)
    .rename(columns={"count": "members", "median": "median family_size of members"})
    .query("members>2")
)
ax = family_size.plot.bar(rot=90)

# TODO: add scatterplot
# -

survived_families = (
    kaggle_train_df.groupby(kaggle_xdata.FamilyName.loc["train"])
    .Survived.agg(["count", "sum"])
    .rename(columns={"sum": "survived"})
    .eval("not_survived=count-survived")
    .query("count>2")
    .sort_values(["count", "survived"], ascending=False)
    .drop(columns=["count"])
)
survived_families_fake = (
    kaggle_train_df.groupby(kaggle_xdata.FamilyName.loc["train"].sample(frac=1).values)
    .Survived.agg(["count", "sum"])
    .rename(columns={"sum": "survived"})
    .eval("not_survived=count-survived")
    .query("count>2")
    .sort_values(["count", "survived"], ascending=False)
    .drop(columns=["count"])
)
axs = create_axs(2, ax_size=(5, 3))
survived_families.plot.bar(stacked=True, ax=next(axs))
survived_families_fake.plot.bar(stacked=True, ax=next(axs), title="Fake")

kaggle_train_df.join(kaggle_xdata.FamilyName.loc["train"]).query(
    "FamilyName == 'Laroche'"
).sort_values("Age", ascending=False)

couples_df = (
    kaggle_data.join(kaggle_xdata)
    # .loc['train']
    .query("PrefixName == 'Mrs' | PrefixName == 'Mr'")
    .pipe(
        lambda df: df.join(
            df.FirstName.str.extract(r"([^\(]*).*")[0].str.strip().rename("HusbandFirstName")
        )
    )
    .groupby(["FamilyName", "HusbandFirstName"])
    .filter(lambda df: len(df) > 1)
)

# +
couples_count = (
    couples_df.groupby(["FamilyName", "HusbandFirstName", "Sex", "Survived"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "not_survived", 1: "survived"})
    .unstack(fill_value=0)
    .value_counts()
    .sort_index()
    .rename("count")
)
couples_count.index = pd.Index(["T?", "?T", "TT", "F?", "FT", "?F", "FF"], name="M/F survived")
couples_count["??"] = (
    len(couples_df.groupby(["FamilyName", "HusbandFirstName"])) - couples_count.sum()
)
couples_count = couples_count.reindex(["TT", "T?", "?T", "FT", "F?", "?F", "FF", "??"])

couples_count.index = pd.MultiIndex.from_tuples(
    [
        ("T", "T"),
        ("T", "O"),
        ("O", "T"),
        ("F", "T"),
        ("F", "O"),
        ("O", "F"),
        ("F", "F"),
        ("O", "O"),
    ],
    names=["Male_Survived", "Female_Survived"],
)
couples_count.unstack(fill_value=0).rename(columns={"O": "?"}, index={"O": "?"})
# -

# ## correlation inside tickets

ticket_group_sizes = kaggle_xdata.groupby("ticket_number").size()
ticket_group_sizes.to_csv("data/ticket_group_sizes.csv")
ticket_group_sizes.value_counts()

survived_groups = (
    kaggle_train_df.groupby(kaggle_xdata.ticket_number.loc["train"])
    .Survived.agg(["count", "sum"])
    .rename(columns={"sum": "survived"})
    .eval("not_survived=count-survived")
    .query("count>2")
    .sort_values(["count", "survived"], ascending=False)
    .drop(columns=["count"])
)
survived_groups_fake = (
    kaggle_train_df.groupby(kaggle_xdata.ticket_number.loc["train"].sample(frac=1).values)
    .Survived.agg(["count", "sum"])
    .rename(columns={"sum": "survived"})
    .eval("not_survived=count-survived")
    .query("count>2")
    .sort_values(["count", "survived"], ascending=False)
    .drop(columns=["count"])
)
axs = create_axs(2, ax_size=(5, 3))
survived_groups.plot.bar(stacked=True, ax=next(axs))
survived_groups_fake.plot.bar(stacked=True, ax=next(axs), title="Fake")
