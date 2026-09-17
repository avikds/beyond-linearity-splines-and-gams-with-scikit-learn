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
- [ ] **7.** extrapolation
- [ ] **8.** smoothing_spline
- [ ] **9.** local_smoother
- [ ] **10.** gam_pipeline
- [ ] **11.** partial_effects
- [ ] **12.** logistic_gam
- [ ] **13.** fit_age_models
- [ ] **14.** test_comparison

---

Built on Deep-ML.
