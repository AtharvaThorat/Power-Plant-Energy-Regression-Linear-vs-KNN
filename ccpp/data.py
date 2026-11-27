"""The UCI Combined Cycle Power Plant data: 9,568 hourly readings from 2006 to 2011 at full load."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PREDICTORS = ["AT", "V", "AP", "RH"]   # ambient temperature (°C), exhaust vacuum (cm Hg),
TARGET = "PE"                          # ambient pressure (mbar), relative humidity (%) -> output (MW)
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "Folds5x2_pp.xlsx"


def load(path=DEFAULT_PATH, sheet="Sheet1"):
    """The workbook holds the same rows shuffled five ways (Sheet1..Sheet5); one sheet is enough."""
    return pd.read_excel(path, sheet_name=sheet)


def split(df, train_size=0.7, seed=42):
    """The 70/30 random split used throughout the main analysis."""
    return train_test_split(df, train_size=train_size, random_state=seed)
