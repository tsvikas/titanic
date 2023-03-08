from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display


def create_axs(n_cols=4, ax_size=(3, 2)):
    fig_size = (ax_size[0] * n_cols, ax_size[1])
    while True:
        fig, axs = plt.subplots(1, n_cols, figsize=fig_size)
        yield from axs


class DataFrameDisplay:
    def __init__(self, index_names=None):
        self.df = None
        # TODO: just auto-get this if dict.
        self.index_names = index_names

    def save(self, filename: Path):
        self.df.to_csv(filename)

    @classmethod
    def load(cls, filename: Path, index_names=None):
        df = pd.read_csv(filename, index_col=index_names)
        df_display = cls(index_names=index_names)
        df_display.df = df
        return df_display

    def add_row(self, data: dict, name):
        if isinstance(name, dict):
            index = [[v] for v in name.values()]
            index_names = list(name.keys())
        elif isinstance(name, list):
            index = [[n] for n in name]
            index_names = self.index_names
        else:
            index = [name]
            index_names = self.index_names
        row = pd.DataFrame(data=data, index=index)
        if index_names is not None:
            row.index.set_names(index_names, inplace=True)
        if self.df is None:
            self.df = row
        else:
            # we want to align indices
            old_index_names = self.df.index.names
            self.df = pd.concat([self.df.reset_index(), row.reset_index()]).set_index(
                old_index_names
            )
        self.display()

    def display(self):
        clear_output(wait=True)
        display(self.df)


def expand_cabin(s):
    return pd.concat(
        [
            (s.str.count(" ").rename("C_count") + 1),
            s.str.slice(0, 1).rename("C_letter"),
            s.str.split(" ", n=1, expand=True)[0]
            .str.slice(1)
            .rename("C_num")
            .replace("", float("nan"))
            .astype(float),
        ],
        axis=1,
    )


def add_features(df, ticket_group_sizes=None):
    features = df[[]]
    features = features.join(expand_cabin(df.Cabin))
    features["family_size"] = df.SibSp + df.Parch
    features["travel_alone"] = features.family_size == 0
    features["ticket_prefix"] = df.Ticket.str.slice(0, 1)
    features["ticket_number"] = df.Ticket.str.extract(r"(.* )?(\d+)")[1].astype("Int64")
    if ticket_group_sizes is not None:
        features["ticket_group_size"] = features["ticket_number"].map(ticket_group_sizes)
    features["is_zero_fare"] = df.Fare == 0
    split_name = df.Name.str.split(",", expand=True, n=1)
    split_prefix_first_name = split_name[1].str.split(".", expand=True, n=1)
    features["FamilyName"] = split_name[0].str.strip()
    features["PrefixName"] = split_prefix_first_name[0].str.strip()
    features["FirstName"] = split_prefix_first_name[1].str.strip()
    return features


# TODO: find better way to find the column names. preferabliy through fit.
kaggle_train_df = pd.read_csv("data/train.csv", index_col=0)
add_features.names = list(kaggle_train_df.pipe(add_features).columns)  # type: ignore[attr-defined]


def compare_col(data, target, col, model=None):
    data_transform = model[:-1].transform(data) if model is not None else data
    col_series = data_transform[col]
    if col_series.dtype == float:
        try:
            col_series = col_series.astype("Int64")
        except TypeError:
            col_series = pd.cut(col_series, 10)
    survived_col = (
        target.groupby(col_series)
        .agg(["count", "sum"])
        .rename(columns={"sum": "survived"})
        .eval("not_survived=count-survived")
        .drop(columns="count")
    )
    return survived_col
