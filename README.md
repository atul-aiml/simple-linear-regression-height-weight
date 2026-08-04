# Simple Linear Regression: Predicting Height from Weight

A from-scratch walkthrough of simple linear regression using `scikit-learn` and `statsmodels`, built to understand the full workflow — not just call `.fit()` and move on.

## 📊 Project Overview

This project predicts a person's **Height** from their **Weight** using a single-feature linear regression model. It's intentionally simple (one predictor) so the focus stays on doing every step correctly: EDA, scaling, train/test splitting, model fitting, and — most importantly — **evaluating the model properly** with multiple metrics.

## 🗂️ Dataset

- `height-weight.csv` — 24 rows, 2 columns (`Weight`, `Height`)
- Small, synthetic-style dataset used for learning purposes, not production modeling

## 🔧 Workflow

1. **EDA** — scatter plot, correlation matrix, pairplot to check the linear relationship visually
2. **Preprocessing** — train/test split (75/25), then `StandardScaler` fit on training data only (to avoid data leakage)
3. **Modeling** — `LinearRegression` from scikit-learn
4. **Evaluation** — MSE, MAE, RMSE, R², Adjusted R²
5. **Cross-check** — refit with `statsmodels.OLS` to inspect the full regression summary (p-values, confidence intervals, F-statistic)

## 📈 Results

| Metric | Value |
|---|---|
| Slope (coefficient) | 17.30 |
| Intercept | 156.47 |
| MAE | 9.67 |
| MSE | 114.84 |
| RMSE | 10.72 |
| R² | 0.736 |
| Adjusted R² | 0.670 |

**Interpretation:** the model explains about 74% of the variance in height using weight alone, with an average prediction error of roughly 10.7 units. R² and Adjusted R² are close here mainly because there's only one predictor — the adjustment penalty barely bites with `k=1`.

⚠️ **Caveat:** the test set is only 6 rows (25% of 24 total), so these metrics are noisy and shouldn't be treated as a robust estimate. This is called out deliberately as a learning point, not glossed over.

## 🧠 Key Things I Learned

- **Why scale after splitting, not before** — fitting the scaler on the full dataset leaks test-set information into training.
- **MAE vs MSE vs RMSE** — when to use each depending on how you want to weight outliers vs. keep interpretable units.
- **R² always non-decreasing with more features, Adjusted R² penalizes that** — why Adjusted R² matters once you're comparing models with different numbers of predictors.
- **statsmodels OLS doesn't add an intercept automatically** — unlike sklearn's `LinearRegression`, you need `sm.add_constant()` on *both* train and test sets, or you'll fit a line forced through the origin and later hit a shape-mismatch `ValueError` at prediction time.

## 🛠️ How to Run

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Practical_Simple_Linear_Regression.ipynb
```

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

## 📌 Next Steps

- Add residual diagnostics (residuals vs. fitted, Q-Q plot) to check regression assumptions
- Try k-fold cross-validation instead of a single train/test split, given the small sample size
- Extend to multiple linear regression with additional features

---

*Built as a hands-on exercise to solidify the fundamentals of linear regression and model evaluation.*
