import pandas as pd


def split_by_time(df):
    matches = (
        df[["match_id", "start_time"]]
        .drop_duplicates()
        .sort_values("start_time")
    )

    n = len(matches)

    train_ids = matches.iloc[:int(n * 0.7)]["match_id"]
    valid_ids = matches.iloc[int(n * 0.7):int(n * 0.85)]["match_id"]
    test_ids = matches.iloc[int(n * 0.85):]["match_id"]

    train = df[df["match_id"].isin(train_ids)].copy()
    valid = df[df["match_id"].isin(valid_ids)].copy()
    test = df[df["match_id"].isin(test_ids)].copy()

    return train, valid, test