B1. Problem Formulation
(a) Machine Learning Formulation
The objective is to predict the number of items sold for each store under different promotional strategies.
Target Variable: items_sold
Input Features:
Store attributes: store size, location type
Promotion details: promotion type
Temporal features: month, festival, weekend
External factors: competition density
This is a supervised learning regression problem because the target variable is continuous (number of items sold). The goal is to learn a function that maps input features to a numerical output.

(b) Why Items Sold Instead of Revenue
Using total revenue as a target variable can be misleading because revenue is influenced by pricing and discount strategies. For example, a heavy discount promotion may increase items sold but reduce total revenue per item.
Items sold is a more reliable measure because it directly reflects customer demand and promotion effectiveness, independent of pricing fluctuations.
This illustrates an important principle in machine learning:
The target variable should align closely with the business objective and should not be distorted by external factors.

(c) Alternative to a Single Global Model
Instead of using a single global model, a better approach is:
Segmented Models (e.g., separate models for urban, semi-urban, and rural stores)
This is justified because:
Customer behavior varies significantly by location
Promotions may perform differently across demographics
Alternatively, a hierarchical or clustered modeling approach can be used where stores with similar characteristics share patterns.



B2. Data and EDA Strategy
(a) Data Joining and Dataset Design
The four tables can be joined as follows:
transactions joined with store attributes using store_id
transactions joined with promotion details using promotion_id
transactions joined with calendar using transaction_date
The grain of the final dataset:
One row per store per day (or per transaction, depending on aggregation level)
Before modeling, aggregation should be performed:
Total items sold per store per day/month
Average basket size
Number of transactions

(b) Exploratory Data Analysis
Key analyses include:
Sales distribution by promotion type
To identify which promotions perform best
Time series plot of items sold
To detect seasonality and trends
Boxplot of items sold by location type
To compare performance across store categories
Correlation heatmap
To identify relationships between features
These insights help in:
Feature selection
Identifying transformations (e.g., log scaling)
Detecting outliers

(c) Handling Promotion Imbalance
Since 80% of data has no promotion:
The model may become biased toward predicting low sales
Promotions may appear less effective than they actually are
To address this:
Use balanced sampling techniques
Introduce feature weighting
Evaluate performance separately on promotion vs non-promotion data



B3. Model Evaluation and Deployment
(a) Train-Test Split and Metrics
The data should be split temporally, using earlier data for training and later data for testing.
Random split is inappropriate because:
It mixes past and future data
Leads to data leakage and unrealistic performance
Evaluation metrics:
RMSE (Root Mean Squared Error): Penalizes large errors
MAE (Mean Absolute Error): Easier to interpret in business terms

(b) Explaining Model Decisions
Feature importance can be used to understand why different promotions are recommended.
For example:
December may show high importance for festival indicators
March may show higher importance for competition density
This helps explain:
Seasonal effects
Customer demand variation
Communicating this to stakeholders builds trust in the model.

(c) Deployment Strategy
Model Saving
Save trained model using joblib or pickle
Monthly Prediction Workflow
Collect new data
Apply same preprocessing pipeline
Generate predictions
Monitoring
Track RMSE/MAE over time
Detect performance drift
Trigger retraining when performance drops
