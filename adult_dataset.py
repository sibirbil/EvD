#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:25:05 2024

@author: u1573378
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_adult_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    df.drop_duplicates(inplace=True)

    # Data cleaning and preparation
    df[df == '?'] = np.nan

    for col in ['workclass', 'occupation', 'native-country']:
        df[col].fillna(df[col].mode()[0], inplace=True)


    # Apply label encoding to the 'education' column
    label_encoder = LabelEncoder()
    df['education'] = label_encoder.fit_transform(df['education'])

    # Remove categorical variables and apply transformations
    columns_to_remove = ['marital-status', 'occupation', 'relationship', 'native-country']
    df.drop(columns=columns_to_remove, inplace=True)

    df['race'] = df['race'].apply(lambda x: 'white' if x.lower() == 'white' else 'others')
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

    # Replace workclass categories and normalize numerical columns
    df['workclass'] = df['workclass'].replace({
        'Private': 'private',
        'Local-gov': 'government',
        'Federal-gov': 'government',
        'State-gov': 'government',
        'Self-emp-not-inc': 'others',
        'Self-emp-inc': 'others',
        'Without-pay': 'Never-worked',
        'Never-worked': 'Never-worked'
    })

    df_encoded = pd.get_dummies(df, columns=['workclass', 'race', 'gender'], drop_first=True)
   
    # scale the large values
    #df_encoded['fnlwgt'] = df_encoded['fnlwgt'] / max(df_encoded['fnlwgt']) 
    #df_encoded['capital-gain'] = df_encoded['capital-gain'] / max(df_encoded['capital-gain'])
    #df_encoded['capital-loss'] = df_encoded['capital-loss'] / max(df_encoded['capital-loss'])

    # Separate features and labels
    X = df_encoded.drop(columns=['income'])
    y = df_encoded['income']

    return X, y, X.columns