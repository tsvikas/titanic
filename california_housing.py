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

# import panel as pn
# pn.extension()
import hvplot.pandas  # noqa
import pandas as pd

housing_df = pd.read_csv("data/housing.csv")
housing_df.head()

# +
# hvplot.explorer(housing_df)
# -

housing_df.hvplot.points("longitude", "latitude", color="ocean_proximity", aspect=1, s=1)

housing_df.hvplot.points(
    "longitude",
    "latitude",
    color="households",
    aspect=1,
    cmap="hot",
    rasterize=True,
    aggregator="sum",
)

housing_df.hvplot.scatter(
    "longitude",
    "latitude",
    color="housing_median_age",
    aspect=1,
    cmap="hot",
    rasterize=True,
    aggregator="mean",
)

housing_df.eval("median_house_value_M = median_house_value/1000000").hvplot.scatter(
    "longitude",
    "latitude",
    color="median_house_value_M",
    aspect=1,
    cmap="plasma",
    rasterize=True,
    aggregator="mean",
)

from titanic.notebook import create_axs

axs = create_axs()
for col in housing_df:
    try:
        housing_df[col].plot.hist(bins=50, ax=next(axs), title=col)
    except TypeError:
        pass

axs = create_axs()
for col in housing_df:
    try:
        housing_df.plot.scatter(
            x=col, y="median_house_value", ax=next(axs), title=col, s=1, alpha=0.5
        )
    except TypeError:
        pass
