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
        index = [[n] for n in name] if isinstance(name, list) else [name]
        row = pd.DataFrame(data=data, index=index)
        if self.index_names is not None:
            row.index.set_names(self.index_names, inplace=True)
        if self.df is None:
            self.df = row
        else:
            self.df = pd.concat([self.df, row])
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


def add_features(df):
    return (
        df[[]]
        .assign(family_size=df.SibSp + df.Parch)
        .assign(travel_alone=lambda df_: (df_.family_size == 0))
        .join(expand_cabin(df.Cabin))
        .join(df.Ticket.str.slice(0, 1).rename("ticket_prefix"))
        .join((df.Fare == 0).rename("is_zero_fare"))
        .join(df.Name.str.split(",", expand=True, n=1)[0].to_frame("FamilyName"))
        .join(
            df.Name.str.split(",", expand=True, n=1)[1]
            .str.split(".", expand=True, n=1)
            .rename(columns={0: "PrefixName", 1: "FirstName"})
        )
        .assign(FamilyName=lambda d: d.FamilyName.str.strip())
        .assign(PrefixName=lambda d: d.PrefixName.str.strip())
        .assign(FirstName=lambda d: d.FirstName.str.strip())
    )


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
