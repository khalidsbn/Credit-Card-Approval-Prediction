""" Build-in functions to use them in data processing """

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def missing_values_table(df, select_numerical=True):
    """
    Generates a table summarizing missing values in numerical or object-type columns of a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame to analyze for missing values.
    - select_numerical (bool): If True, analyze missing values in numerical columns.
                              If False, analyze missing values in object-type columns.

    Returns:
    - mis_val_table_ren_columns (pd.DataFrame): A DataFrame summarizing missing values with columns:
        - 'Missing Values': The count of missing values in each selected column.
        - '% of Total Values': The percentage of missing values relative to the total number of values.
    """
    if select_numerical:
        columns = df.select_dtypes('number').columns
    else:
        columns = df.select_dtypes('object').columns

    # Total missing values
    mis_val = df[columns].isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df[columns].isnull().sum() / len(df)

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


def imbalance_feature_checker(df, select_numerical=True):
    """
    Checks for imbalance in categorical or numerical features of a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame containing features to be checked for imbalance.
    - select_numerical (bool): If True, check for imbalance in numerical columns.
                              If False, check for imbalance in object-type columns.

    Returns:
    - imbalance_df (pd.DataFrame): A DataFrame containing the imbalance percentages for each selected column.
    """
    # Filter columns based on data type
    if select_numerical:
        columns = df.select_dtypes('number').columns
    else:
        columns = df.select_dtypes('object').columns

    # Create an empty DataFrame to store the imbalance percentages
    imbalance_df = pd.DataFrame(columns=['Value'])

    # Loop through selected columns in df
    for column in columns:
        # Calculate the value counts as a percentage
        value_counts_percentage = df[column].value_counts(normalize=True)

        # Check if the most common value occurs more than 88% of the time
        if len(value_counts_percentage) > 0 and value_counts_percentage.iloc[0] > 0.88:
            # Append the imbalance percentage to imbalance_df with the column name as the index
            imbalance_df.loc[column] = value_counts_percentage.iloc[0]

    # Print some summary information
    print("Your selected dataframe has " + str(len(columns)) + " columns.\n"
          "There are " + str(imbalance_df.shape[0]) +
          " columns that are imbalance.")

    return imbalance_df


def feature_deletor(df, cols):
    """
    Deletes specified columns from a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame from which columns are to be deleted.
    - cols (list of str): The list of column names to be deleted.

    Returns:
    - df (pd.DataFrame): The DataFrame with the specified columns removed.
    """
    return df.drop(cols, axis=1)


def convert_to_year(df, cols):
    """
    Converts specified columns representing days to years in a pandas DataFrame by taking the ceiling of the division by -365.25.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame containing the columns to be converted.
    - cols (list of str): The list of column names representing days to be converted to years.

    Returns:
    - df (pd.DataFrame): The DataFrame with the specified columns converted to years.
    """
    for i in cols:
        df[i] = np.ceil(df[i] / -365.25)
    return df


def log_transform_feature(df, feature_list, epsilon=1e-10):
    """
    Applies a natural logarithm transformation to specified numerical features in a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame containing the features to be transformed.
    - feature_list (list of str): The list of feature names to be log-transformed.
    - epsilon (float): A small value added to features before applying the logarithm to handle zeros.

    Returns:
    - df (pd.DataFrame): The DataFrame with the specified features log-transformed.
    """
    for feature in feature_list:
        df[feature] = np.log(df[feature] + epsilon)
    return df


def fill_nan(df, select_numerical=True):
    """
    Fills missing values in a pandas DataFrame based on data type.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with missing values.
    - select_numerical (bool): If True, fill missing values in numerical columns using the mean.
                              If False, fill missing values in object-type columns using the mode.

    Returns:
    - df (pd.DataFrame): The DataFrame with missing values filled.
    """
    if select_numerical:
        cols = [col for col in df.select_dtypes('number').columns if df[col].isna().sum() != 0]
        for i in cols:
            df[i] = df[i].fillna(df[i].mean())
    else:
        cols = [col for col in df.select_dtypes('object').columns if df[col].isna().sum() != 0]
        for i in cols:
            df[i] = df[i].fillna(df[i].mode().iloc[0])
    return df


def encode_ordinal_features(df, ordinal_encoder_feature):
    """
    Encodes ordinal features in a pandas DataFrame using scikit-learn's OrdinalEncoder.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame containing the features to be encoded.
    - ordinal_encoder_feature (str or list of str): The name or names of the ordinal feature(s) to be encoded.

    Returns:
    - df (pd.DataFrame): The DataFrame with the specified ordinal feature(s) encoded.
    """
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_encoder_feature] = ordinal_encoder.fit_transform(df[[ordinal_encoder_feature]])
    return df



def unique_values_counter(df):
    """
    Counts the number of unique values in each column of a pandas DataFrame for object-type columns.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame for which unique values need to be counted.

    Returns:
    - unique_counts (pd.Series): A pandas Series containing the count of unique values for each object-type column.
    """
    object_columns = df.select_dtypes('object')
    unique_counts = object_columns.apply(pd.Series.nunique, axis=0)
    return unique_counts # pd.Series


# Use it to ckeck if there is any common features between merging two dataframes
def common_features_checker(df_1, df_2):
    """
    Finds and returns a list of common columns between two pandas DataFrames.

    Parameters:
    - df_1 (pd.DataFrame): The first pandas DataFrame.
    - df_2 (pd.DataFrame): The second pandas DataFrame.

    Returns:
    - common_cols (list): A list containing the names of columns that are common to both DataFrames.
    """
    if not isinstance(df_1, pd.DataFrame) or not isinstance(df_2, pd.DataFrame):
        raise ValueError("Both inputs should be pandas DataFrames.")
    common_cols = list(set(df_1.columns) & set(df_2.columns))
    return common_cols
