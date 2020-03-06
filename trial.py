import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


def cat_cols(df):
    """ This function takes in a dataframe and spits out the categorical columns"""
    cat_cols_bool = (df.dtypes == 'object')  # An interim object before cat_cols (columns with categorical data)
    return set(cat_cols_bool[cat_cols_bool].index)


def column_analyser(df):
    """ This function takes in a dataframe and analyses it to see which columns we want to drop and tells us which
    the different types are. It is split into different conditions that we're looking for."""

    # Columns that are categorical with high cardinality
    high_cat_cols = {col for col in cat_cols(df) if df[col].nunique() > 10}

    # Columns with too many NaN values
    col_nans = (df.isnull().sum())
    super_nan_cols = set(col_nans[col_nans > 40].index)

    return set(df.columns) - (high_cat_cols.union(super_nan_cols))


def my_ordinal_encoder(df, categories):
    """ Takes in a dataframe and encodes it in an ordinal way as specified by the categories list. """

    # Checking if there are rogue elements (things that are neither nan nor in the categories list) in the dataframe
    # that will confuse the encoder.
    for col in df:
        rogue_series = ~df[col].isin(categories + [np.nan])
        if rogue_series.sum() >= 1:
            print("DataFrame and categories list mismatching in " + col)
            return 0

    # Encoder
    categories_codes = [categories.index(categories[i]) for i in range(0, len(categories))]
    dictionary_encoder = dict(zip(categories, categories_codes))
    nan_code = {np.nan: np.nan}
    dictionary_encoder.update(nan_code)
    for col in df.columns:
        df[col + '_enc'] = df.loc[:, col].apply(lambda x: dictionary_encoder[x])
        df = df.drop(col, axis=1)

    return df

""" Main """
# Data paths
houses_train_data_full_path = "~/datasets/houses_data/train.csv"
houses_test_data_path = "~/datasets/houses_data/test.csv"

# Reading in the data
train_data_full = pd.read_csv(houses_train_data_full_path)
test_data_full = pd.read_csv(houses_test_data_path)


# Dropping columns that cause problems, intersecting training and test for consistency. Target dropped here.
train_rem_cols = column_analyser(train_data_full)
test_rem_cols = column_analyser(test_data_full)
rem_cols = train_rem_cols.intersection(test_rem_cols)
X_remcolled = train_data_full[rem_cols]
y = train_data_full[target]
test_data = test_data_full[rem_cols]

# Columns which requiring encoding (with categories given for ordinal encoding)
ordcols = [{'ExterQual', 'ExterCond', 'KitchenQual'},
           {'Functional'}]
cats = [['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']]
ohcols = cat_cols(X_remcolled) - set.union(*ordcols)

df = my_ordinal_encoder(X_remcolled[ordcols[0]], cats[0])
