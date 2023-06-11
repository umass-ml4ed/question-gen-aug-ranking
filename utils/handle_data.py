"""
python -m code.utils.create_dataset_split 

Only for first creation:
python -m code.utils.create_dataset_split --first_creation
"""

import pandas as pd
import os
import numpy as np

RAW_DIR = "./data/"

def load_df(filename, folder, nrows=None):
    filename = os.path.join(folder, filename)
    df = pd.read_csv(filename, nrows=nrows)
    df = df.fillna("")
    return df

def save_csv(df, filename, dirname):
    filepath = os.path.join(dirname, filename + ".csv")
    df.to_csv(filepath, encoding='utf-8', index=False)