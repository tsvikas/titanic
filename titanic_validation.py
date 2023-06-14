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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from tqdm import trange
from xgboost import XGBClassifier

from titanic.model import build_model

# -


SPLIT_SEED = 0

# ## load the data

input_dir = Path("/kaggle/input/titanic")
if not input_dir.exists():
    input_dir = Path("data")

kaggle_train_df = pd.read_csv(input_dir / "train.csv", index_col=0)
kaggle_test_df = pd.read_csv(input_dir / "test.csv", index_col=0)

kaggle_train_df.sample(5).sort_index()

# ## learn

# ### split

# +
# add validation for final estimation of the model
train_df, val_df = train_test_split(
    kaggle_train_df, test_size=100, stratify=kaggle_train_df.Survived, random_state=SPLIT_SEED
)
train_target = train_df.pop("Survived")
val_target = val_df.pop("Survived")

len(train_df), len(val_df)
# -


# ### create best model

# +
best_params = {"learning_rate": 0.5, "max_depth": 2, "n_estimators": 100}
model = build_model(classifier=XGBClassifier(**best_params))
model.fit(train_df, train_target)

features_df = pd.Series(
    model[-1].feature_importances_, index=model[-1].feature_names_in_
).sort_values(ascending=False)
features_df.head(10).plot.bar()
# -

# ### finalize_and_predict


model = model.set_params(classifier__n_jobs=-1)


def train_once(model, n_train, data=kaggle_train_df):
    x_train, x_test = train_test_split(data, train_size=n_train)
    y_train = x_train.pop("Survived")
    y_test = x_test.pop("Survived")
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    n_correct = (pred == y_test).sum()
    n_wrong = len(y_test) - (pred == y_test).sum()
    return n_correct, n_wrong, n_correct / (n_correct + n_wrong)


# +
def train_and_score(model, n_train):
    n_correct = n_wrong = 0
    while n_correct + n_wrong < 1000:
        n_correct0, n_wrong0, _accuracy = train_once(model, n_train)
        n_correct += n_correct0
        n_wrong += n_wrong0
    return n_correct, n_wrong


train_and_score(model, 500)
# -

# ## run on different n_train size

scores = {i: train_and_score(model, i) for i in trange(100, 500)}

pd.DataFrame.from_dict(scores, "index", columns=["correct", "wrong"]).plot()

res300 = pd.DataFrame(
    [train_and_score(model, 300) for i in trange(100)], columns=["correct", "wrong"]
)

res300.eval("correct/(correct+wrong)").mean()

res300.eval("correct/(correct+wrong)").std()

res300.eval("correct/(correct+wrong)").plot.hist()

import numpy as np

# +
ax = None
ax = (
    pd.DataFrame.from_dict(scores, "index", columns=["correct", "wrong"])
    .eval("correct/(correct+wrong)")
    .plot(linewidth=0, marker=".")
)

model_selection.LearningCurveDisplay.from_estimator(
    model,
    kaggle_train_df.drop(columns="Survived"),
    kaggle_train_df["Survived"],
    ax=ax,
    train_sizes=np.linspace(0.1, 1.0, 20),
    n_jobs=-1,
    shuffle=True,
)
ax.plot(300, res300.eval("correct/(correct+wrong)").mean(), ".r")
# -

scores2 = []

for i in trange(50):
    for n in [100, 200, 300, 400, 500, 600, 700]:
        scores2.append([n, *train_once(model, n)])

scores2_df = pd.DataFrame(scores2, columns=["n_train", "correct", "wrong", "accuracy"])
scores2_df.plot.scatter("n_train", "accuracy")

import seaborn as sns

len(kaggle_train_df)

sns.relplot(data=scores2_df, x="n_train", y="accuracy", kind="line", errorbar="sd")

scores3 = []

data = kaggle_train_df.sample(n=600)

for i in trange(50):
    for n in [100, 200, 300, 400, 500]:
        scores3.append([n, *train_once(model, n, data=data)])

scores3_df = pd.DataFrame(scores3, columns=["n_train", "correct", "wrong", "accuracy"])
fig, ax = plt.subplots()
# ax = scores3_df.plot.scatter("n_train", "accuracy")
sns.lineplot(data=scores3_df, x="n_train", y="accuracy", errorbar="sd", ax=ax, label="600")
sns.lineplot(data=scores2_df, x="n_train", y="accuracy", errorbar="sd", ax=ax, label="891")

(
    so.Plot(scores3_df, x="n_train", y="accuracy")
    .add(so.Line(), so.Agg())
    .add(so.Band(), so.Est(errorbar="sd"))
)


import seaborn.objects as so

pd.concat({600: scores3_df, 891: scores2_df}, names=["n_total"]).reset_index(0).reset_index(
    drop=True
).n_total.unique()

(
    so.Plot(
        pd.concat({600: scores3_df, 891: scores2_df}, names=["n_total"])
        .reset_index(0)
        .reset_index(drop=True),
        x="n_train",
        y="accuracy",
        color="n_total",
    )
    .scale(color=so.Nominal())
    .add(so.Line(), so.Agg())
    .add(so.Band(), so.Est(errorbar="sd"))
)
