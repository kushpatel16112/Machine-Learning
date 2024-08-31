import pandas as pd
import numpy as np

df = pd.read_csv("../Files/cleaned_file.csv")


def replace_outliers_with_mean(df, columns, threshold=3):
    """
    Identify outliers in specified columns using Z-score and replace them with the mean of the column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to check for outliers.
    threshold (float): The Z-score threshold for defining outliers. Default is 3.

    Returns:
    pd.DataFrame: A DataFrame with outliers replaced by the column mean.
    """
    for column in columns:
        # Calculate the Z-scores for each data point in the column
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        
        # Identify outliers
        outliers = z_scores > threshold
        
        # Replace outliers with the mean of the column
        df.loc[outliers, column] = df[column].mean()
    
    return df


# Specify the columns you want to check for outliers
columns_to_check = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

# Replace outliers with the mean in the specified columns
df_cleaned = replace_outliers_with_mean(df, columns_to_check)

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('cleaned_file_outlier.csv', index=False)
