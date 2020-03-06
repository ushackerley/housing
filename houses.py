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


def total_encoder(df1, df2, ordcols, cats, ohcols):
    """ Encodes the input dataframes with ordinal and one hot information. It then matches columns between the two,
    which one hot encoding can ruin, given that some categories don't occur in both. """

    df = [df1, df2]
    enc = [None] * 2

    # Encoded dataframes
    for j in range(0, len(df)):
        enc[j] = [my_ordinal_encoder(df[j][list(ordcols[i])], cats[i]) for i in range(0, len(ordcols))]
        enc[j] = enc[j] + [pd.get_dummies(df[j][list(ohcols)])]
        df[j] = df[j].drop(cat_cols(df[j]), axis=1)
        df[j] = pd.concat([df[j]] + enc[j], axis=1)

    # Match up the columns with zeros post one hot encoding, since this creates columns in one and not the other.
    if len(df[0].columns) > len(df[1].columns):
        i = 0
        j = 1
    elif len(df[0].columns) < len(df[1].columns):
        i = 1
        j = 0
    else:
        return df

    missing_cols = set(df[i].columns) - set(df[j].columns)
    missing_cols_df = pd.DataFrame(np.zeros((df[j].shape[0], len(missing_cols))), columns=list(missing_cols))
    df[j] = pd.concat([df[j], missing_cols_df], axis=1)

    return df


""" Main """
# Data paths
houses_train_data_full_path = "~/datasets/houses_data/train.csv"
houses_test_data_path = "~/datasets/houses_data/test.csv"

# Reading in the data
train_data_full = pd.read_csv(houses_train_data_full_path)
test_data_full = pd.read_csv(houses_test_data_path)

# Dropping NaN rows if occurring in target column
target = 'SalePrice'
identifier = 'Id'
train_data_full = train_data_full.dropna(subset=[target])

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

# Encoding.
X_encoded, test_encoded = total_encoder(X_remcolled, test_data, ordcols, cats, ohcols)

# Imputation of train and test data, replacing column headings.
my_imputer = SimpleImputer()
X = pd.DataFrame(my_imputer.fit_transform(X_encoded))
X.columns = X_encoded.columns
test_imputed = pd.DataFrame(my_imputer.fit_transform(test_encoded))

# Splitting the training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Decision Tree Model internal
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X_train, y_train)
val_predictions = dt_model.predict(X_valid)
val_mae = mean_absolute_error(val_predictions, y_valid)
print("The MAE for the dt_model is ", val_mae)
print("Predictions look like: ", val_predictions[0:5], '\n')

# Random Forest Model internal
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)
val_predictions = rf_model.predict(X_valid)
val_mae = mean_absolute_error(val_predictions, y_valid)
print("The MAE for the rf_model is ", val_mae)
print("Predictions look like: ", val_predictions[0:5], '\n')

# Creating a full RFM with imputed data
rf_model_full = RandomForestRegressor(random_state=1)
rf_model_full.fit(X, y)
val_predictions = rf_model_full.predict(test_imputed)

# Creating an output CSV file
val_pred_series = pd.Series(val_predictions, name=target)
final_predictions = pd.concat([test_data[identifier], val_pred_series], axis=1)
final_predictions = final_predictions.set_index(identifier)
final_predictions.to_csv('./my_predictions.csv')
