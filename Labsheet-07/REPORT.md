# Lab Sheet 06 – Report

## Workflow used in both projects
Load -> inspect (missing, duplicates) -> clean -> train/test split (80/20, fixed seed) -> preprocessing **inside a Pipeline**
(imputation, scaling, one-hot encoding, fitted on training data only to avoid leakage) -> compare several algorithms with
test metrics and 5-fold cross-validation -> visualise -> save with Joblib -> reload and predict a new record.

## Project 1: Student Performance Prediction
Inputs: attendance, internal marks, assignment marks, study hours/day, previous semester %. 500 students after cleaning.

**Regression (final score 0–100)**

| Model | MAE | RMSE | R² test | R² CV |
|---|---|---|---|---|
| Linear Regression | 4.39 | 5.45 | 0.883 | 0.865 |
| Random Forest | 5.05 | 6.42 | 0.838 | 0.828 |
| Gradient Boosting | 5.14 | 6.60 | 0.829 | 0.824 |
| Decision Tree | 5.97 | 7.88 | 0.757 | 0.678 |

Linear Regression wins: the target is close to a linear combination of the inputs, and CV R² ≈ test R² shows no overfitting.
The average error is about 4 marks.

**Classification (Pass if score ≥ 40; 432 pass / 68 fail)**: Logistic Regression, KNN and Random Forest all reach ~93–94% accuracy.
Random Forest has the best F1 (0.966). Because Fail is the minority class, check the Fail recall (0.71) in the classification report,
not accuracy alone – missed at-risk students are the costly error in an early-warning system.

## Project 2: House Price Prediction (price in lakh INR)
Inputs: area, bedrooms, bathrooms, location (categorical), parking, age. 800 houses after cleaning.

| Model | MAE | RMSE | R² test | R² CV |
|---|---|---|---|---|
| Gradient Boosting | 6.23 | 8.10 | 0.941 | 0.941 |
| Linear Regression | 6.38 | 8.56 | 0.935 | 0.937 |
| Ridge / Lasso | 6.38 | 8.54 / 8.53 | 0.935 | 0.937 |
| Random Forest | 7.09 | 8.78 | 0.931 | 0.929 |
| Decision Tree | 9.82 | 12.37 | 0.863 | 0.830 |

GridSearchCV tuning of Gradient Boosting (learning_rate 0.1, max_depth 2, 200 trees) improved it to
**MAE 5.67, RMSE 7.48, R² 0.950**, so it is the saved final model (`model/house_price_model.joblib`).
Area and location are the most important features; age lowers the price.

## Viva questions
1. Why split into train and test? (to estimate performance on unseen data)
2. Why put imputation/scaling inside a Pipeline? (prevents data leakage; one object to save/deploy)
3. R² vs MAE vs RMSE? (variance explained / average error / error penalising outliers)
4. Why not accuracy alone for Pass/Fail? (class imbalance – look at precision, recall, F1)
5. Ridge vs Lasso? (L2 shrinks coefficients, L1 can set them to zero)
6. What does cross-validation tell you? (stability / overfitting of the model)
7. Why one-hot encode location? (no false ordering between categories)
