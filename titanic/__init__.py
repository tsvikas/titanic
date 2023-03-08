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
        .join(expand_cabin(df.Cabin))
        .join(df.Ticket.str.slice(0, 1).rename("ticket_prefix"))
        .join((df.Fare.fillna(1) < 1e-16).rename("is_zero_fare"))
        .join((df.eval("Parch+SibSp").fillna(1) < 1e-16).rename("travel_alone"))
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


def compare_col(data, col, model=None):
    data_transform = model[:-1].transform(data) if model is not None else data
    col_series = data_transform[col]
    if col_series.dtype == float:
        try:
            col_series = col_series.astype("Int64")
        except TypeError:
            col_series = pd.cut(col_series, 10)
    survived_col = (
        data.groupby(col_series)
        .Survived.agg(["count", "sum"])
        .rename(columns={"sum": "survived"})
        .eval("not_survived=count-survived")
        .drop(columns="count")
    )
    return survived_col
