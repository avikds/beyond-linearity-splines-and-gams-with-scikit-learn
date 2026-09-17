# Beyond Linearity: Splines and GAMs with Scikit-Learn

Chapter 7 of An Introduction to Statistical Learning on the book's own Wage data, with the tools a practitioner reaches for. Fit wage against age with polynomials, step functions from KBinsDiscretizer, regression splines from SplineTransformer, a penalized smoothing spline with its effective degrees of freedom, and a local nearest-neighbor smoother; see why a degree-four polynomial explodes past the data while a spline with linear extrapolation does not; then assemble a generalized additive model with a ColumnTransformer that gives age and year their own spline terms and education a set of dummies, read its partial effects, turn it into a logistic GAM for high earners, and finish with one test-set table that compares every age-only model at its cross-validated setting.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_wage
- [x] **2.** split_wage
- [x] **3.** cv_tools
- [x] **4.** polynomial_regression
- [x] **5.** step_functions
- [x] **6.** spline_regression
- [x] **7.** extrapolation
- [x] **8.** smoothing_spline
- [x] **9.** local_smoother
- [x] **10.** gam_pipeline
- [x] **11.** partial_effects
- [x] **12.** logistic_gam
- [x] **13.** fit_age_models
- [x] **14.** test_comparison

## Results

```
Wage data: n=3000, age 18-80, mean wage 111.7k, 5 education levels
train 2250 / test 750; the test set is opened once, at the end

polynomial degree, CV MSE: d1=1609  d2=1538  d3=1535  d4=1536  d5=1538  d6=1540  (fold SE about 94: the curve is flat, CV minimum at degree 3)
  nested F-tests: d2: F=104.7 p=0.0000  d3: F=5.2 p=0.0220  d4: F=1.6 p=0.2100  d5: F=0.0 p=0.9543  -> ANOVA keeps degree 3
step functions, CV MSE by bins: 2=1657  4=1572  8=1541  16=1548  -> one-SE 4 bins, edges [18.0, 33.5, 49.0, 64.5, 80.0]

cubic regression spline, CV MSE by knots: 3=1536  4=1539  5=1541  6=1542  8=1539  12=1541  -> one-SE 3 knots
beyond the data at ages [70, 80, 90, 100]:
  degree-4 polynomial : [106.6, 84.7, 39.6, -45.3]  (range 151.9)
  spline, constant tail: [107.3, 81.1, 81.1, 81.1]
  spline, linear tail  : [107.3, 81.1, 43.4, 5.7]

penalized spline (20 knots): alpha=0.001: df=21.5  alpha=0.1: df=20.3  alpha=1: df=19.1  alpha=10: df=14.7  alpha=100: df=6.6  alpha=1000: df=1.9
  CV minimum at alpha=10; one-SE rule picks alpha=100
local smoother: CV minimum at span 0.1, one-SE picks 0.4; roughness of the curve at span 0.02 = 5.928, at 0.4 = 0.302

GAM wage ~ s(age) + s(year) + education: 15 basis columns; CV MSE 1200 vs age-only spline 1536
  partial-effect ranges: education 61.2, age 40.8, year 6.5 (thousand dollars)
  education effects: < HS Grad -27.4, HS Grad -16.1, Some College -2.3, College Grad +11.9, Advanced Degree +33.8
  logistic GAM for wage > 250k (55 of 2250 workers): P(high) at ages 25..65 = [0.0017, 0.0029, 0.0056, 0.005, 0.004], CV AUC 0.81

test-set RMSE (thousand dollars):
  spline   rmse= 42.14 setting=3
  poly     rmse= 42.17 setting=3
  step     rmse= 42.72 setting=4
  smooth   rmse= 42.98 setting=100.0
  local    rmse= 43.12 setting=0.4
  linear   rmse= 43.34 setting=1
  gam      rmse= 37.30 setting=age+year+education
fitted wage at ages 20/40/60/80: linear [96, 110, 123, 137], poly [71, 117, 115, 96], spline [68, 117, 116, 88], smooth [98, 113, 115, 110]
```
