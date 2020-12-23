from pandas import DataFrame


def find_columns_with_value(df: DataFrame, value_to_find: any):
    col_matched = []
    for c in df.columns:
        df = df[df[c].isin([value_to_find])]
        if df.shape[0] > 0:
            col_matched.append(c)

    return col_matched