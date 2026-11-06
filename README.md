# Delhivery Last-Mile Delay Intelligence Pipeline

## Data findings

### 1. Delay rate by service type

Express shipments experience the highest delay rate at **48.97% (0.4897)**, followed by Standard shipments at **20.56% (0.2056)** and Economy shipments at **8.57% (0.0857)**. This indicates that Express shipments require particular operational attention.

### 2. Delay rate across hub tiers

The delay rate increases progressively from **4.00% in Tier 1** to **21.50% in Tier 2** and **44.00% in Tier 3**. This suggests that lower-tier hubs face considerably greater last-mile delivery challenges.

### 3. Delay rate across distance bands

Distance has a strong relationship with delivery delays. The delay rate increases from **2.19% (0–200 km)** to **13.68% (201–400 km)**, **43.64% (401–600 km)** and **85.23% (601+ km)**. Longer delivery distances are associated with substantially higher delay rates.

### 4. Features correlated with delivery hours

The three features with the strongest absolute correlations with delivery hours are:

- `promised_hours`: **+0.8372**
- `distance_km`: **+0.4013**
- `n_stops`: **+0.3247**

All three show positive correlations with delivery hours, with promised delivery hours having the strongest linear relationship.

### 5. Declared-value transformation

The original declared-value column has a mean of **INR 1,533.0542** and a median of **INR 1,121.50**. After applying the natural logarithm, the mean becomes **7.0277** and the median becomes **7.0224**. The much smaller difference between the transformed mean and median indicates that the logarithmic transformation reduces the influence of high-value observations and makes the distribution less dominated by larger values.

## Model results (regression)

### 1. Baseline performance

The mean baseline achieved a validation R² of **-0.0212**. Its negative R² indicates that predicting the mean delivery time from the training set performs worse than using the validation-set mean as a constant prediction. This baseline is therefore not useful for accurate delivery-time estimation.

### 2. Model comparison

Four regression algorithms were evaluated: Linear Regression, Decision Tree Regressor, Random Forest Regressor and Gradient Boosting Regressor.

**Linear Regression** achieved the lowest validation RMSE of **4.8912 hours** and an R² of **0.9313**, making it the selected final model according to the validation RMSE selection rule.

Gradient Boosting performed almost equally well, with a validation RMSE of **4.9165 hours** and an R² of **0.9306**.

### 3. Decision tree depth analysis

The depth-1 decision tree exhibits high bias because both training RMSE (**11.6334**) and validation RMSE (**12.1324**) are high. The model is too simple to capture the underlying relationships.

The unlimited-depth tree exhibits high variance, with training RMSE of **0.0000** and validation RMSE of **7.8368**. It memorizes the training data but generalizes less effectively to unseen observations.

The best validation performance was obtained at depth **6**, with an RMSE of **6.2796 hours**.

### 4. Gradient Boosting learning-rate experiment

A learning rate of **0.1** produced the best validation performance, with an R² of **0.9306**.

A learning rate of **0.01** was too slow to learn effectively with 100 estimators, resulting in an R² of **0.7127**. A learning rate of **1.0** produced overly aggressive updates and weaker generalization, resulting in an R² of **0.8440**.

### 5. Cross-validation and final model selection

Cross-validation produced a mean R² of **0.9261** for Linear Regression and **0.9353** for Gradient Boosting. Therefore, the cross-validation ranking differed from the validation ranking.

Despite Gradient Boosting having a higher cross-validated mean R², Linear Regression was selected because the prescribed selection rule requires choosing the model with the lowest validation RMSE. Its validation RMSE of **4.8912 hours** was slightly lower than Gradient Boosting's **4.9165 hours**.

The selected Linear Regression model achieved a test RMSE of **4.5235 hours** and a test R² of **0.9493**.

## Model results (classification)

### 1. Blind baseline

The blind baseline achieved an accuracy of **0.7667**, but its precision, recall and F1-score were all zero. It predicted every parcel as on-time and failed to identify any of the 42 delayed parcels in the validation set. Therefore, despite its apparently reasonable accuracy, it is operationally useless.

### 2. Feature importance and noise analysis

The seven strongest features according to ANOVA F-score were `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`.

The five bookkeeping noise columns collectively contributed **0.2245 (22.45%)** of the Random Forest's feature importance. However, all five had F-ranks outside the top seven and individual F-scores below 10. This demonstrates that Random Forest can assign some importance to features that have weak individual class-separation signals.

### 3. SelectKBest experiment

The experiment compared 5, 7, 10, 12 and 17 selected features. Selecting seven features achieved the highest validation F1-score of **0.8333**. Therefore, seven features were retained for subsequent model tuning and final classification.

### 4. Hyperparameter tuning

Grid Search achieved a best cross-validation accuracy of **0.8952** and validation F1 of **0.7750**. Randomized Search achieved a cross-validation accuracy of **0.8905** and validation F1 of **0.7848**.

The untuned Random Forest using the same seven selected features achieved the highest validation F1 of **0.8333**.

This demonstrates that tuning does not guarantee improved performance. The validation dataset contains only 180 observations, so small changes in predictions can noticeably affect the F1-score. The untuned configuration was therefore selected.

### 5. Alert threshold selection

The selected alert threshold was **0.5**, the highest threshold that maintained validation recall of at least 0.80.

At this threshold, validation precision was **0.7955** and recall was **0.8333**. Raising the threshold to 0.6 increased precision to 0.8889 but reduced recall to 0.7619, violating the operational requirement.

### 6. Final test performance

The final classifier achieved test precision of **0.6889**, recall of **0.7561** and an F1-score of **0.7209**.

This means approximately 6.89 out of every 10 flagged parcels were genuinely delayed, while approximately 7.56 out of every 10 genuinely delayed parcels were identified.

Test recall was lower than validation recall of 0.8333. This difference should be interpreted carefully because the test dataset contains only 41 delayed parcels, making the recall estimate sensitive to a small number of classification errors.