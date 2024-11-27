
"""
Created on Wed Nov 27 17:37:38 2024

@author: u1573378
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def prepare_credit_dataset(file_path, iqr_multiplier=2):
  

    df = pd.read_csv(file_path)
    iqr_multiplier = 2

   # Impute missing values using the median for all input columns
    imputer = SimpleImputer(strategy='median')
    input_cols = df.columns  # All feature columns
    df[input_cols] = imputer.fit_transform(df[input_cols])


    # Remove outliers for relevant columns
    df = remove_outliers(df, 'DebtRatio', iqr_multiplier)
    df = remove_outliers(df, 'MonthlyIncome', iqr_multiplier)

    # Split the dataset into features (X) and target (y)
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])

    return X, y


def remove_outliers(df, column_name, iqr_multiplier = 2):
    # TODO check this iqr multiplier default value if it is too stringent
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    df = df[(df[column_name] >= lower_bound) & \
            (df[column_name] <= upper_bound)]
    return df






