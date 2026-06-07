import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Map a 2-class object/boolean series to 0/1.
    Known maps:
      Yes/No -> 1/0
      Male/Female -> 1/0
    Fallback: first seen -> 0, second -> 1 (stable order).
    """
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # known cases
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # generic 2-class map
    if len(vals) == 2:
        mapping = {vals[0]: 0, vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # not binary -> return as-is (will be one-hot encoded)
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Encode features:
    - for object cols with exactly 2 distinct values -> map to 0/1
    - for object cols with >2 values -> one-hot (drop_first=True)
    - keep numeric as-is
    - convert bool dtype to int
    """
    df = df.copy()

    # find categorical (object) excluding target
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]

    # split into binary / multi
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    # binary maps
    for c in binary_cols:
        df[c] = _map_binary_series(df[c].astype(str))

    # booleans to int (if any)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # one-hot for multi-category
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # after mapping, some binary cols may be Int64 (nullable); cast to int
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    return df

