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