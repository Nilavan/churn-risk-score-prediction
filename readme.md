# Churn risk score prediction

[HackerEarth machine learning challenge:](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-customer-churn/?utm_source=challenges-modern&utm_campaign=participated-challenges&utm_medium=right-panel) How NOT to lose a customer in 10 days

## About challenge

### Problem

Churn rate is a marketing metric that describes the number of customers who leave a business over a specific time period. Every user is assigned a prediction value that estimates their state of churn at any given time. This value is based on:

- User demographic information
- Browsing behavior
- Historical purchase data among other information

It factors in our unique and proprietary predictions of how long a user will remain a customer. This score is updated every day for all users who have a minimum of one conversion. The values assigned are between 1 and 5.

### Task

To predict the churn score for a website based on the features provided in the dataset.

### Data description

| First Header | Second Header |
| ------------ | ------------- |
| Content Cell | Content Cell  |
| Content Cell | Content Cell  |

### Evaluation metric

score = 100 x metrics.f1_score(actual, predicted, average="macro")
