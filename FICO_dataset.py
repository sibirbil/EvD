#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:00:36 2024

@author: u1573378
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

file_path = ".data/FICO_dataset.csv"
df = pd.read_csv(file_path)

X = df.drop(columns = 'RiskPerformance')
y = df.RiskPerformance.replace(to_replace=['Bad', 'Good'], value=[1, 0])