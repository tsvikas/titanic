from pathlib import Path
from typing import Any, Generator

import pandas as pd
from IPython.core.display_functions import clear_output, display
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.pipeline import Pipeline
from typing_extensions import Self


def create_axs(n_cols: int = 4, ax_size: tuple[int, int] = (3, 2)) -> Generator[Axes, None, None]:
    fig_size = (ax_size[0] * n_cols, ax_size[1])
    while True:
        fig, axs = plt.subplots(1, n_cols, figsize=fig_size)
        yield from axs


class DataFrameDisplay:
    def __init__(self, index_names: list[str] | None = None):
        self.df: pd.DataFrame | None = None
        self.index_names = index_names

    def save(self, filename: Path) -> None:
        assert self.df is not None
        self.df.to_csv(filename)

    @classmethod
    def load(cls, filename: Path, index_names: list[str] | None = None) -> Self:
        df = pd.read_csv(filename, index_col=index_names)
        df_display = cls(index_names=index_names)
        df_display.df = df
        return df_display

    def add_row(self, data: dict[str, Any], name: dict[str, Any] | list[Any] | Any) -> None:
        index: list[Any] | list[list[Any]]
        index_names: list[str] | None
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
            if all(name is not None for name in self.df.index.names):
                # we want to align indices
                old_index_names = self.df.index.names
                self.df = pd.concat([self.df.reset_index(), row.reset_index()]).set_index(
                    old_index_names
                )
            else:
                self.df = pd.concat([self.df, row])
        self.display()

    def display(self) -> None:
        clear_output(wait=True)
        display(self.df)


def compare_col(
    data: pd.DataFrame,
    target: pd.Series,
    col: str,
    model: Pipeline | None = None,
) -> pd.DataFrame:
    data_transform = model[:-1].transform(data) if model is not None else data
    col_series = data_transform[col]
    if col_series.dtype == float:
        try:
            col_series = col_series.astype("Int64")
        except TypeError:
            col_series = pd.cut(col_series, 10)
    survived_col: pd.DataFrame = (
        target.groupby(col_series)
        .agg(["count", "sum"])
        .rename(columns={"sum": "survived"})
        .eval("not_survived=count-survived")
        .drop(columns="count")
    )
    return survived_col
