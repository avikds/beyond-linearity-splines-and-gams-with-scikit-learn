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

# Step 3 - cv_tools
from sklearn.model_selection import cross_val_score

def cv_mse(model, X, y, cv):
    # cross_val_score returns negative MSE because sklearn's scoring
    # convention treats larger scores as better.
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error"
    )

    # Convert negative MSE scores into positive MSE values.
    fold_mse = -scores

    # Compute the mean MSE and its standard error across folds.
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(mean_mse, 1), round(se_mse, 1)

def cv_curve(make_model, X, y, values, cv):
    # Evaluate each hyperparameter/value using cross-validated MSE.
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)

        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses

def one_se_rule(values, means, ses, prefer="smaller"):
    # Identify the setting with the lowest cross-validated mean MSE.
    best_index = int(np.argmin(means))
    best_mean = means[best_index]
    best_se = ses[best_index]

    # The one-standard-error threshold is measured from the best mean.
    threshold = best_mean + best_se

    # Keep all settings whose mean error is within one SE of the best.
    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    elif prefer == "larger":
        return max(eligible)
    else:
        raise ValueError("prefer must be either 'smaller' or 'larger'")

# Step 4 - polynomial_regression
from scipy.stats import f as f_dist

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def poly_model(degree):
    # Scale age before generating polynomial features so that higher powers
    # remain numerically well-behaved.
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    )

def poly_curve(X, y, degrees, cv):
    # Evaluate polynomial degrees using the cross-validation helper
    # implemented in Step 3.
    return cv_curve(poly_model, X, y, degrees, cv)

def anova_degrees(X, y, max_degree):
    # Compare each degree-d polynomial with the nested degree-(d-1)
    # polynomial using the standard partial F-test.
    n = len(y)
    results = []

    for d in range(2, max_degree + 1):
        model_prev = poly_model(d - 1)
        model_curr = poly_model(d)

        model_prev.fit(X, y)
        model_curr.fit(X, y)

        # Residual sum of squares for the two nested models.
        residual_prev = y - model_prev.predict(X)
        residual_curr = y - model_curr.predict(X)

        rss_prev = np.sum(residual_prev ** 2)
        rss_curr = np.sum(residual_curr ** 2)

        # Since the two models differ by one parameter, the numerator
        # degrees of freedom of the F-test is 1.
        F = ((rss_prev - rss_curr) / 1) / (rss_curr / (n - d - 1))

        # Survival function gives P(F_{1, n-d-1} >= observed F).
        p_value = f_dist.sf(F, 1, n - d - 1)

        results.append(
            (d, round(F, 2), round(p_value, 4))
        )

    return results

def choose_degree(X, y, degrees, cv, alpha=0.05):
    # Select the degree with the lowest cross-validated MSE.
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    # Perform the nested polynomial F-tests up to the largest requested
    # degree. The ANOVA selection stops at the first non-significant test.
    max_degree = max(degrees)
    anova_results = anova_degrees(X, y, max_degree)

    degree_anova = 1

    for d, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = d
        else:
            break

    return degree_min, degree_anova

def curve_on_grid(model, grid):
    # Generate predictions on the supplied age grid and return them as
    # a one-dimensional NumPy array rounded to two decimal places.
    predictions = model.predict(grid)

    return np.round(np.asarray(predictions).ravel(), 2)

# Step 5 - step_functions
from sklearn.preprocessing import KBinsDiscretizer

def step_model(n_bins):
    # Discretize age into equally spaced bins and fit a linear regression
    # to the resulting one-hot encoded step indicators.
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform"
        ),
        LinearRegression()
    )

def step_curve(X, y, bins, cv):
    # Evaluate the candidate numbers of bins using cross-validated MSE.
    return cv_curve(step_model, X, y, bins, cv)

def choose_bins(X, y, bins, cv):
    # Compute cross-validated mean MSE for each candidate bin count.
    means, _ = step_curve(X, y, bins, cv)

    # Choose the bin count with the minimum cross-validated MSE.
    bins_min = bins[int(np.argmin(means))]

    # Apply the one-standard-error rule, preferring the smaller
    # number of bins.
    bins_1se = one_se_rule(bins, means, _, prefer="smaller")

    return bins_min, bins_1se

def bin_edges(model):
    # Retrieve the fitted bin edges from the KBinsDiscretizer.
    discretizer = model.named_steps["kbinsdiscretizer"]
    edges = discretizer.bin_edges_[0]

    return [round(float(edge), 1) for edge in edges]

def step_levels(model, grid):
    # Predict the fitted step-function values on the supplied grid.
    predictions = model.predict(grid)

    # Round the predictions first, then remove duplicates and sort them.
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))

    return [float(level) for level in levels]

# Step 6 - spline_regression
from sklearn.preprocessing import SplineTransformer

def spline_model(n_knots, degree=3, extrapolation="constant"):
    # Build a regression-spline model:
    # SplineTransformer -> LinearRegression.
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False
        ),
        LinearRegression()
    )

def spline_basis_size(model, X):
    # Get the fitted spline transformer from the pipeline.
    transformer = model.named_steps["splinetransformer"]

    # Transform a few rows and use the resulting matrix width to determine
    # the number of spline basis columns.
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])

    return transformed.shape[1]

def spline_curve(X, y, knot_counts, cv):
    # Evaluate the candidate knot counts using cross-validated MSE.
    return cv_curve(spline_model, X, y, knot_counts, cv)

def choose_knots(X, y, knot_counts, cv):
    # Compute cross-validated means and standard errors for each knot count.
    means, ses = spline_curve(X, y, knot_counts, cv)

    # Select the knot count with the lowest cross-validated MSE.
    k_min = knot_counts[int(np.argmin(means))]

    # Apply the one-standard-error rule, preferring fewer knots.
    k_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")

    return k_min, k_1se

# Step 7 - extrapolation
def beyond_data(models, ages):
    # Build a one-column DataFrame containing the requested ages.
    grid = pd.DataFrame({"age": ages})

    # Generate rounded predictions for each fitted model.
    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }

def extrapolation_report(X, y, ages):
    # Fit the degree-four polynomial model.
    poly4 = poly_model(4).fit(X, y)

    # Fit a cubic regression spline with constant extrapolation.
    spline_const = spline_model(5).fit(X, y)

    # Fit the same spline with linear extrapolation.
    spline_linear = spline_model(
        5,
        extrapolation="linear"
    ).fit(X, y)

    # Collect predictions beyond the observed data range.
    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_const,
            "spline_linear": spline_linear
        },
        ages
    )

    # Compute the range of the degree-four polynomial predictions.
    poly4_range = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1
    )

    predictions["poly4_range"] = poly4_range

    return predictions

# Step 8 - smoothing_spline
from sklearn.linear_model import Ridge

def smooth_model(alpha, n_knots=20):
    # Build a penalized regression-spline model:
    # SplineTransformer -> Ridge regression.
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False
        ),
        Ridge(alpha=alpha)
    )

def effective_df(model, X):
    # Retrieve the fitted spline transformer and ridge estimator.
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    # Construct the spline basis matrix.
    B = transformer.transform(X)

    # Center each basis column before computing the effective degrees
    # of freedom. The intercept is handled separately as unpenalized.
    B = B - B.mean(axis=0)

    # Compute the effective degrees of freedom of the penalized spline:
    # 1 + tr((B'B + alpha I)^(-1) B'B)
    alpha = ridge.alpha
    n_basis = B.shape[1]

    BtB = B.T @ B
    penalty = BtB + alpha * np.eye(n_basis)

    edf = 1.0 + np.trace(
        np.linalg.inv(penalty) @ BtB
    )

    return round(float(edf), 2)

def smooth_curve(X, y, alphas, cv):
    # Evaluate the candidate smoothing penalties using cross-validation.
    return cv_curve(smooth_model, X, y, alphas, cv)

def choose_alpha(X, y, alphas, cv):
    # Compute cross-validated MSE and standard errors for each alpha.
    means, ses = smooth_curve(X, y, alphas, cv)

    # Select the alpha with the minimum cross-validated MSE.
    alpha_min = alphas[int(np.argmin(means))]

    # Apply the one-standard-error rule, preferring the larger alpha
    # because it corresponds to stronger smoothing.
    alpha_1se = one_se_rule(
        alphas,
        means,
        ses,
        prefer="larger"
    )

    return alpha_min, alpha_1se

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

