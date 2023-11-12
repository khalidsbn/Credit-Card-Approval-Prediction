
"""
missing_values_table(X, select_numerical=True)

imbalance_feature_checker(X, select_numerical=True)

feature_deletor(X, cols)

convert_to_year(X, cols)

log_transform_feature(X, feature_list)

fill_nan(X, select_numerical=True)

encode_ordinal_features(X, ordinal_encoder_feature)

unique_values_counter(X)

common_features_checker(X1, X2)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def missing_values_table(X, select_numerical=True):
    if select_numerical:
        columns = X.select_dtypes('number').columns
    else:
        columns = X.select_dtypes('object').columns

    # Total missing values
    mis_val = X[columns].isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * X[columns].isnull().sum() / len(X)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(len(columns)) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    return mis_val_table_ren_columns


def imbalance_feature_checker(X, select_numerical=True):
    # Filter columns based on data type
    if select_numerical:
        columns = X.select_dtypes('number').columns
    else:
        columns = X.select_dtypes('object').columns

    # Create an empty DataFrame to store the imbalance percentages
    imbalance_df = pd.DataFrame(columns=['Value'])

    # Loop through selected columns in df
    for column in columns:
        # Calculate the value counts as a percentage
        value_counts_percentage = X[column].value_counts(normalize=True)

        # Check if the most common value occurs more than 88% of the time
        if len(value_counts_percentage) > 0 and value_counts_percentage.iloc[0] > 0.88:
            # Append the imbalance percentage to imbalance_df with the column name as the index
            imbalance_df.loc[column] = value_counts_percentage.iloc[0]

    # Print some summary information
    print("Your selected dataframe has " + str(len(columns)) + " columns.\n"
          "There are " + str(imbalance_df.shape[0]) +
          " columns that are imbalance.")

    return imbalance_df


# Define the function to delete columns with NaN values
def feature_deletor(X, cols):
    return X.drop(cols, axis=1)


def convert_to_year(X, cols):
  for i in cols:
    X[i] = np.ceil(X[i] / -365.25)
  return X


def log_transform_feature(X, feature_list, epsilon=1e-10):
    for feature in feature_list:
        X[feature] = np.log(X[feature] + epsilon)
    return X


def fill_nan(X, select_numerical=True):
  if select_numerical:
    cols = [col for col in X.select_dtypes('number').columns if X[col].isna().sum() != 0]
    for i in cols:
      X[i] = X[i].fillna(X[i].mean())
  else:
    cols = [col for col in X.select_dtypes('object').columns if X[col].isna().sum() != 0]
    for i in cols:
      X[i] = X[i].fillna(X[i].mode().iloc[0])
  return X


def encode_ordinal_features(X, ordinal_encoder_feature):
    ordinal_encoder = OrdinalEncoder()
    X[ordinal_encoder_feature] = ordinal_encoder.fit_transform(X[ordinal_encoder_feature])
    return X


# calcule the number of unique values of object features
def unique_values_counter(X):
    object_columns = X.select_dtypes('object')
    unique_counts = object_columns.apply(pd.Series.nunique, axis=0)
    return unique_counts # pd.Series


# check if there are any common features between two datasets before merge them.
def common_features_checker(X1, X2):
    if not isinstance(X1, pd.DataFrame) or not isinstance(X2, pd.DataFrame):
        raise ValueError("Both inputs should be pandas DataFrames.")
    common_cols = list(set(X1.columns) & set(X2.columns))
    return common_cols



