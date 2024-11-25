import pandas as pd

# Helper function to handle static feature processing
def handle_static_feature_data(df, column_name, feature_list, feature_prefix):
    if df.empty:
        feat_df = pd.DataFrame(0, index=[0], columns=feature_list)
    else:
        df['val'] = 1
        df = df.drop_duplicates().pivot(columns=column_name, values='val').fillna(0)
        feat_df = pd.DataFrame(0, index=[0], columns=list(set(feature_list) - set(df.columns)))
        df = pd.concat([df, feat_df], axis=1)[feature_list]
    df.columns = pd.MultiIndex.from_product([[feature_prefix], df.columns])
    return df

# Helper function to add missing indices and fill data
def add_missing_indices(df, los):
    add_indices = pd.Index(range(los)).difference(df.index)
    add_df = pd.DataFrame(index=add_indices, columns=df.columns).fillna(np.nan)
    df = pd.concat([df, add_df]).sort_index().fillna(0)
    return df

# Helper function for imputing missing values
def impute_values(df, method):
    if method == 'Mean':
        df = df.ffill().bfill().fillna(df.mean())
    elif method == 'Median':
        df = df.ffill().bfill().fillna(df.median())
    return df.fillna(0)

# Helper function to handle dynamic feature processing
def handle_feature_data(df, column_name, feature_list, feature_prefix, los, impute_method=None):
    if df.empty:
        val = pd.DataFrame(0, index=range(los), columns=feature_list)
        val.columns = pd.MultiIndex.from_product([[feature_prefix], val.columns])
        return val

    val = df.pivot_table(index='start_time', columns=column_name, values='dose_val_rx')
    add_missing_indices(val, los)
    val = impute_values(val, impute_method)

    feat_df = pd.DataFrame(0, index=range(los), columns=list(set(feature_list) - set(val.columns)))
    val = pd.concat([val, feat_df], axis=1)[feature_list]
    val.columns = pd.MultiIndex.from_product([[feature_prefix], val.columns])
    return val