# import libraries

from os import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV


def preprocess(df_train, df_test):
    '''Function to preprocess data
    1. Delete/update possibly incorrect values
    2. Drop rows with churn risk score = -1
    3. Fill NaNs
    '''
    df_train.avg_frequency_login_days = abs(pd.to_numeric(
        df_train.avg_frequency_login_days, errors='coerce'))

    l5 = np.where(df_train['churn_risk_score'] == -1)
    df_train.drop(l5[0], inplace=True)

    df_train.medium_of_operation = df_train.medium_of_operation.fillna(
        value='Both')
    df_train.gender = df_train.gender.fillna('F')
    df_train.avg_time_spent = abs(df_train.avg_time_spent)

    df_test = pd.read_csv('./dataset/test.csv',
                          na_values=['xxxxxxxx', 'Unknown', '?', -999])
    df_test.avg_frequency_login_days = abs(pd.to_numeric(
        df_test.avg_frequency_login_days, errors='coerce'))
    df_test.medium_of_operation = df_test.medium_of_operation.fillna(
        value='Both')
    df_test.gender = df_test.gender.fillna('F')
    df_test.avg_time_spent = abs(df_test.avg_time_spent)

    cols_nan = ['region_category', 'joined_through_referral', 'preferred_offer_types',
                'days_since_last_login', 'avg_frequency_login_days', 'points_in_wallet']

    for col in cols_nan:
        if df_train[col].dtype == 'float64':
            df_train[col] = df_train[col].fillna(df_train[col].mean())
            df_test[col] = df_test[col].fillna(df_test[col].mean())
        else:
            df_train[col] = df_train[col].fillna(method='ffill')
            df_test[col] = df_test[col].fillna(method='ffill')

    df_train.points_in_wallet = abs(df_train.points_in_wallet)
    df_test.points_in_wallet = abs(df_test.points_in_wallet)

    return df_train, df_test


def join_date(df_train, df_test):
    '''Function to create columns
    -> joining date to datetime
    -> joining year
    -> joining month
    -> joining day
    -> last visit time to datetime
    -> diff = total days
    '''
    df_train.joining_date = pd.to_datetime(df_train.joining_date)
    df_train['joining_year'] = df_train.joining_date.dt.year
    df_train['joining_month'] = df_train.joining_date.dt.month
    df_train['joining_day'] = df_train.joining_date.dt.day
    df_train.last_visit_time = pd.to_datetime(df_train.last_visit_time)
    # total days = diff
    df_train['diff'] = ((df_train['last_visit_time'] - df_train['joining_date']
                         ).apply(lambda x: str(x).split()[0])).astype('float64')

    df_test.joining_date = pd.to_datetime(df_test.joining_date)
    df_test['joining_year'] = df_test.joining_date.dt.year
    df_test['joining_month'] = df_test.joining_date.dt.month
    df_test['joining_day'] = df_test.joining_date.dt.day
    df_test.last_visit_time = pd.to_datetime(df_test.last_visit_time)
    df_test['diff'] = ((df_test['last_visit_time'] - df_test['joining_date']
                        ).apply(lambda x: str(x).split()[0])).astype('float64')

    return df_train, df_test


def label_encode(df_train, df_test):
    '''Function to label encode selected columns'''
    le = preprocessing.LabelEncoder()

    le_cols = ['gender', 'used_special_discount', 'offer_application_preference',
               'past_complaint', 'joined_through_referral', 'membership_category', 'feedback']

    for col in le_cols:
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.fit_transform(df_test[col])

    return df_train, df_test


def one_hot(df_train, df_test):
    '''Function to one hot encode selected columns'''
    one_hot_cols = ['region_category', 'preferred_offer_types',
                    'medium_of_operation', 'internet_option', 'complaint_status']

    df_train = pd.get_dummies(df_train, columns=one_hot_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=one_hot_cols, drop_first=True)

    return df_train, df_test


def drop_cols(df_train, cols_to_drop):
    '''Function to drop selected columns'''
    df_train.drop(cols_to_drop, axis=1, inplace=True)
    return cols_to_drop


def final_model(x, y, x_train, y_train, x_cv, y_cv):

    # Cross validation to find best params
    '''params={'learning_rate':[0.05,0.1,0.15,0.2],
        'max_depth':[3,4,5,6,8,10],
        'min_child_weight':[1,3,5,7],
        'gamma':[0.0,0.1,0.2,0.3,0.4,0.5],
        'colsample_bytree':[0.3,0.4,0.5,0.7]}
    rsearch=RandomizedSearchCV(xgc,param_distributions=params,n_iter=100,cv=3,random_state=42,n_jobs=-1)
    rsearch.fit(x,y)
    rsearch.best_parameter_ '''

    xgb_model = make_pipeline(RobustScaler(), XGBClassifier(
        colsample_bytree=0.7, gamma=0.4, max_depth=4, min_child_weight=3, n_estimators=119, objective='multi:softprob'))
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_cv)

    print(classification_report(y_cv, y_pred))
    print(f1_score(y_cv, y_pred, average='macro'))
    xgb_model.fit(x, y)
    result = xgb_model.predict(x_test)

    return result


def make_submission(df_test, result):
    submission = pd.DataFrame({
        'customer_id': df_test['customer_id'],
        'churn_risk_score': result,
    })
    submission.to_csv('sub.csv', index=False)


if __name__ == "__main__":
    df_train = pd.read_csv('./dataset/train.csv',
                           na_values=['xxxxxxxx', 'Unknown', '?', -999])
    df_test = pd.read_csv('./dataset/test.csv',
                          na_values=['xxxxxxxx', 'Unknown', '?', -999])
    df_train, df_test = preprocess(df_train, df_test)
    df_train, df_test = join_date(df_train, df_test)
    df_train, df_test = label_encode(df_train, df_test)
    df_train, df_test = one_hot(df_train, df_test)

    cols_to_drop = ['customer_id', 'Name', 'security_no',
                    'referral_id', 'joining_date', 'last_visit_time']
    drop_cols(df_train, cols_to_drop)

    # train cv split
    x = df_train.drop(['churn_risk_score'], axis=1)
    y = df_train['churn_risk_score']
    x_test = df_test.drop(cols_to_drop, axis=1)
    x_train, x_cv, y_train, y_cv = train_test_split(
        x, y, train_size=0.7, stratify=y)

    result = final_model(x, y, x_train, y_train, x_cv, y_cv)
    make_submission(df_test, result)
