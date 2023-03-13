import pandas as pd


def expand_cabin(s: pd.Series) -> pd.DataFrame:
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


def add_features(
    df: pd.DataFrame, ticket_group_sizes: dict[int, int] | None = None
) -> pd.DataFrame:
    features = df[[]]
    features = features.join(expand_cabin(df.Cabin))
    features["family_size"] = df.SibSp + df.Parch
    features["travel_alone"] = features.family_size == 0
    features["ticket_prefix"] = df.Ticket.str.slice(0, 1)
    features["ticket_number"] = df.Ticket.str.extract(r"(.* )?(\d+)")[1].astype("Int64")
    if ticket_group_sizes is not None:
        features["ticket_group_size"] = features["ticket_number"].map(ticket_group_sizes)
    features["is_zero_fare"] = df.Fare == 0
    split_name = df["Name"].str.split(",", expand=True, n=1)
    split_prefix_first_name = split_name[1].str.split(".", expand=True, n=1)
    features["FamilyName"] = split_name[0].str.strip()
    features["PrefixName"] = split_prefix_first_name[0].str.strip()
    features["FirstName"] = split_prefix_first_name[1].str.strip()
    return features
