"""
Beyond Linearity: Splines and GAMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_wage
import os
import tempfile
import urllib.request
import pandas as pd

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"

def load_wage():
    # Store the downloaded Wage.csv in the system temporary directory.
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    # Download the dataset only if it is not already present.
    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    # Load and return the dataset as a pandas DataFrame.
    return pd.read_csv(path)

def describe_wage(df):
    # Return the requested summary statistics in the specified format.
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (int(df["age"].min()), int(df["age"].max())),
        "wage_mean": round(df["wage"].mean(), 2),
        "n_education_levels": df["education"].nunique(),
    }

# Step 2 - split_wage
import numpy as np
from sklearn.model_selection import train_test_split

def split_wage(df, test_size=0.25, random_state=0):
    # Split the Wage data into training and test sets.
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    return train, test

def age_xy(df):
    # Use age as a one-column feature DataFrame and wage as the target.
    X = df[["age"]]
    y = df["wage"]

    return X, y

def age_grid(lo=18, hi=80, n=63):
    # Create an evenly spaced grid of ages for plotting fitted curves.
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })

# Step 3 - cv_tools (not yet solved)
# TODO: implement

# Step 4 - polynomial_regression (not yet solved)
# TODO: implement

# Step 5 - step_functions (not yet solved)
# TODO: implement

# Step 6 - spline_regression (not yet solved)
# TODO: implement

# Step 7 - extrapolation (not yet solved)
# TODO: implement

# Step 8 - smoothing_spline (not yet solved)
# TODO: implement

# Step 9 - local_smoother (not yet solved)
# TODO: implement

# Step 10 - gam_pipeline (not yet solved)
# TODO: implement

# Step 11 - partial_effects (not yet solved)
# TODO: implement

# Step 12 - logistic_gam (not yet solved)
# TODO: implement

# Step 13 - fit_age_models (not yet solved)
# TODO: implement

# Step 14 - test_comparison (not yet solved)
# TODO: implement

