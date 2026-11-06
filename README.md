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

## How to run

### Prerequisites

- Python 3.12
- Git
- The four Python dependencies pinned in `requirements.txt`

### Clone the repository

```bash
git clone https://github.com/SwaroopSam2910/Delhivery-delay-intelligence.git

cd Delhivery-delay-intelligence
```

### Create and activate the virtual environment

For Windows using Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Execute the pipeline in order

Run every script from the repository root:

```bash
python generate_data.py

python part_a_clean_eda.py

python part_b_regression.py

python part_c_classification.py

python part_d_segments.py

python package_model.py

python predict_delay.py
```

The generator produces 24 hubs, 1,224 raw historical shipment rows and 48 new shipment rows. The generated dataset files are included in the repository and can be regenerated using the provided generator.

The scripts must be executed in the above order because later stages depend on the outputs of earlier stages.

### Model artefact compatibility

The trained classifier is saved as `models/delay_model.pkl`.

Running `python package_model.py` regenerates this model along with its threshold, feature-order and imputation configuration files.

The committed pickle is designed to be loaded using the scikit-learn version pinned in `requirements.txt`. Pickled scikit-learn models are not guaranteed to be compatible across different versions. If the installed version differs, the model should be regenerated by running `package_model.py` in the target environment rather than relying on the committed pickle.

All scripts use local repository files and free Python libraries. No paid external service or API is required.

## Operations recommendation

### 1. Hubs requiring escalation

The K-Means Escalate segment contains nine hubs with the following historical delay rates:

| Hub | City | Historical delay rate |
|---|---|---:|
| HUB13 | Nagpur | 46% |
| HUB15 | Kochi | 36% |
| HUB17 | Patna | 46% |
| HUB18 | Ranchi | 38% |
| HUB19 | Guwahati | 62% |
| HUB20 | Raipur | 48% |
| HUB21 | Bhubaneswar | 40% |
| HUB23 | Siliguri | 44% |
| HUB24 | Agartala | 54% |

These hubs should receive the Escalate operational playbook, including closer hub-manager supervision, investigation of route constraints and attention to rider capacity and dispatch operations.

HUB19 Guwahati has the highest historical delay rate in this segment at 62%.

### 2. Tier-versus-segment mismatches

Four hubs operate differently from what their assigned tier suggests:

| Hub | City | Assigned tier | Actual segment |
|---|---|---:|---|
| HUB07 | Kolkata | Tier 1 | Watch |
| HUB13 | Nagpur | Tier 2 | Escalate |
| HUB15 | Kochi | Tier 2 | Escalate |
| HUB22 | Dehradun | Tier 3 | Watch |

The predefined tier should therefore be treated as a reference classification rather than a definitive measure of operating performance. Actual operational decisions should also consider the measured delivery behaviour of each hub.

### 3. Delay alert policy

The selected classification threshold is **0.5**.

On the final test dataset, the classifier achieved precision of **0.6889** and recall of **0.7561**.

This means approximately **6.9 out of every 10 flagged parcels are genuinely delayed**, while approximately **7.6 out of every 10 genuinely delayed parcels are caught**.

The validation recall was 0.8333, compared with test recall of 0.7561. This difference is important when considering how the system may perform on future shipments.

### 4. New shipment action list

The prediction pipeline processed all 48 incoming shipments:

| Operational outcome | Shipments |
|---|---:|
| Total shipments | 48 |
| Flagged shipments | 8 |
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| No action | 40 |

The classifier's delay probability is combined with the hub's operating segment to determine the action.

A flagged shipment in the Escalate segment is assigned to the hub manager. A flagged shipment in another segment receives a proactive customer call. An unflagged shipment requires no action.

### 5. Why three hub segments?

K-Means with three clusters produced an across-to-within distance ratio of **5.3928**, with cluster sizes of **9-8-7**.

The ratio increased to **6.3047** for four clusters and **6.8866** for five clusters. However, the four-cluster solution contains a cluster of only four hubs, while the five-cluster solution contains clusters of two and three hubs.

These smaller clusters would be difficult to support with independent operational playbooks. Therefore, three segments provide a practical balance between cluster separation and operational manageability.

## Production plan

### 1. Production readiness

The project addresses the following production-readiness characteristics:

| Characteristic | Implementation in this repository |
|---|---|
| Accuracy | `outputs/part_b_metrics.csv` and `outputs/part_c_final_test.csv` document regression and classification test performance. |
| Reliability | `outputs/part_d_reload_check.csv` verifies that the reloaded model produces identical probabilities to the in-memory model. |
| Reproducibility | `generate_data.py`, pinned dependencies, fixed random states and repeatable prediction output support reproducible execution. |
| Performance | `predict_delay.py` loads a previously trained model instead of retraining it for every incoming shipment batch. |
| Maintainability | The project separates data preparation, regression, classification, segmentation, model packaging and prediction into individual scripts. |
| Consistency | Training and prediction reuse `clean_shipments()` and `build_features()` from `src/shipment_prep.py`. Historical imputation medians and the exact feature order are stored as model artefacts. |
| Monitorability | The final action CSV records shipment identifiers, hub segments, delay probabilities, flags and operational actions. Model evaluation CSVs provide additional monitoring evidence. |

### 2. Operational monitoring

The following are proposed initial production monitoring triggers. They should be reviewed and calibrated once sufficient real operational history becomes available.

| Quantity to monitor | Monitoring frequency | Proposed trigger |
|---|---|---|
| Actual delayed shipment share | Weekly, after delivery outcomes are known | Investigate if the rate exceeds the historical non-sale baseline of 23.17% by more than 5 percentage points for two consecutive weeks. |
| Share of shipments flagged by the classifier | Weekly | Investigate if more than 25% of shipments are flagged for two consecutive weeks. |
| Realised classifier recall | Rolling four-week period, after outcomes mature | Review the model and alert threshold if recall falls below 80%. |
| Feature distribution drift | Weekly | Investigate if the Population Stability Index exceeds 0.20 for an important model input. |

These triggers are intended to identify deterioration early rather than automatically change the model or its threshold without investigation.

### 3. Model retraining schedule

The proposed retraining schedule is quarterly, using newly completed shipment records with verified delivery outcomes.

An earlier review or retraining cycle should be considered when monitoring identifies sustained deterioration in recall, changes in the shipment population or significant feature drift.

The retrained model should undergo the same data preparation, model evaluation, threshold selection and reload verification procedures before deployment.

### 4. Handling sale events and data drift

The synthetic dataset identifies a sale-event window from 15 September to 28 September using `sale_event_flag`.

Sale periods should not be compared directly with ordinary operating weeks because promotional activity can change shipment volume and delivery behaviour.

Performance during a sale event should instead be compared with equivalent promotional periods from previous events. The `sale_event_flag` should remain part of the model input.

Data drift could appear as changes in distance distributions, parcel volumes, service-type proportions, number of rider stops, hub delay rates or the frequency of sale-event shipments.

These changes should be monitored over time. A substantial shift in the input distributions or a deterioration in realised model performance should trigger investigation and potentially retraining.

## Limitations

1. **Synthetic dataset:** The entire project uses synthetic shipment records. Although the generator represents operational patterns, its results do not establish how the model would perform on actual Delhivery shipment data.

2. **Small test dataset:** The classification test set contains only 41 delayed shipments. Consequently, a small number of additional missed or correctly identified shipments can noticeably change the measured recall.

3. **Single validation split:** Model selection and alert threshold selection use one prescribed stratified validation split. Performance may differ with another sample or future shipment population. A temporal evaluation on real historical data would provide stronger evidence of generalisation.

4. **Uncalibrated probabilities:** The classifier's predicted probabilities have not undergone a separate probability-calibration assessment. They should not automatically be interpreted as perfectly calibrated real-world delay probabilities.

5. **Limited operational features:** The model uses the features provided by the synthetic generator. Live traffic, vehicle availability, rider attendance, route disruptions and other external operational conditions are not explicitly represented.

6. **Operational thresholds require review:** The selected alert threshold is based on the prescribed validation recall requirement. Actual operational costs of missed delays and unnecessary customer calls should be assessed before using the system for real-world decisions.