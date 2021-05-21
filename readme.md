# Churn risk score prediction <a name="top"></a>

[HackerEarth machine learning challenge:](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-customer-churn/?utm_source=challenges-modern&utm_campaign=participated-challenges&utm_medium=right-panel) How NOT to lose a customer in 10 days

## Contents

- [Problem](#problem)
- [Task](#task)
- [Data description](#data_desc)
- [Evaluation metric](#metric)
- [Steps](#steps)
- [Ideas](#ideas)
- [Areas to improve](#improve)
- [Final submission](#sub)

## Problem <a name="problem"></a>

Churn rate is a marketing metric that describes the number of customers who leave a business over a specific time period. Every user is assigned a prediction value that estimates their state of churn at any given time. This value is based on:

- User demographic information
- Browsing behavior
- Historical purchase data among other information

It factors in our unique and proprietary predictions of how long a user will remain a customer. This score is updated every day for all users who have a minimum of one conversion. The values assigned are between 1 and 5.

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>

## Task <a name="task"></a>

To predict the churn score for a website based on the features provided in the dataset.

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>

## Data description <a name="data_desc"></a>

The dataset folder contains the following files:

- train.csv: 36992 x 25
- test.csv: 19919 x 24

See the columns in the dataset [here](data_description.md)

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>

## Evaluation metric <a name="metric"></a>

**score** = 100 x metrics.f1_score(actual, predicted, average="macro")

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>

## Steps <a name="steps"></a>

- [x] Load data
- [x] Preprocess data
- [x] Perform [exploratory data analysis](churn_risk_eda.ipynb)
- [x] Feature engineer
- [x] Build and test [different models](churn_risk_models_test.ipynb)
- [x] Make predictions using best model (XGBoost)
- [x] [Submit](final.py)

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>

## Ideas <a name="ideas"></a>
1. NaNs in gender column replaced with F based on customer names
2. Rows with churn risk score = -1 removed
    - Trial 1. Found correlation of all columns with churn risk score column
    - Noticed that replacing -1 score with 4 had best correlation
    - Trial 2. Removing rows with -1 score gives best model accuracy
3. NaNs in medium of operation replaced with 'both' (increased correlation with churn risk score)
4. Columns had incorrect negative values which were converted to positive
    - avg_time_spent
    - points_in_wallet
    - avg_frequency_login_days
6. NaNs for other columns were filled with mean in case of float datatype and ffill method otherwise
7. Values in columns joining_date and last_visit_time were converted to datetime
8. Created new columns (Increased model f1 score)
    - joining_year
    - joining_month
    - joining_day
    - diff (total days)
9. Label encoding for columns
    - gender
    - used_special_discount
    - offer_application_preference
    - past_complaint
    - joined_through_referral
    - membership_category
    - feedback
10. One hot encoding for columns
    - region_category
    - preferred_offer_types
    - medium_of_operation
    - internet_option
    - complaint_status
11. Dropped unnecessary columns
12. Tried various oversampling techniques as churn risk scores 1 and 2 had very few data points compared to 3, 4, and 5
13. Tried various models and found that xgboost and random forest models worked best with the former having an edge

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>

## Areas to improve <a name="improve"></a>

- 

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>

## Final submission <a name="sub"></a>

- Online score: 76.76408
- Offline score: 76.64014
- Rank: [61](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-customer-churn/leaderboard/predict-the-churn-risk-rate-11-fb7a760d/page/2/)

<div align="right">
    <b><a href="#">&#x261D back to top</a></b>
</div>
