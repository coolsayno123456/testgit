# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:02:36 2022

@author: 86151
"""

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import random
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier,AdaBoostClassifier
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import learning_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

import joblib




# 处理缺失值的函数
def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].values
    known_age = age_df[age_df.Age.notnull()].values

    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictedages = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedages
    return df, rfr


def set_cabin_type(df):
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'NO'
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'YES'
    return df

def data_processing(train_data):
    # 对文本进行处理变成数字
    dummies_Cabin = pd.get_dummies(train_data['Cabin'], prefix='Cabin') #我记得好像是做了个onehot
    dummies_Embarked = pd.get_dummies(train_data['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(train_data['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(train_data['Pclass'], prefix='Pclass')
    df = pd.concat([train_data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # pd.set_option('display.max_columns',None)  #用于设置输出所有列名
    # print(df)
    # 对跨度大的数据缩放
    scaler = preprocessing.StandardScaler()
    age_scale = scaler.fit(df[['Age']])
    df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale)
    fare_scale = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale)

    return df, scaler, age_scale, fare_scale

def create_train_cv_set(train_split,cv_split):
    # 构建训练矩阵和cv矩阵
    train_df = train_split.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    cv_df = cv_split.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    cv_np = cv_df.values
    return train_np, cv_np

def train_model(clf,train_np,cv_np):
    train_y = train_np[:, 0]
    train_X = train_np[:, 1:]
    cv_y = cv_np[:,0]
    cv_X = cv_np[:,1:]
    clf.fit(train_X,train_y)
    score_train = clf.score(train_X,train_y)
    score_cv = clf.score(cv_X,cv_y)
    return score_train, score_cv, clf

def list_score(list_train_score, list_cv_score, model, train_np,cv_np):
    score_tr, score_cv, model = train_model(model,train_np,cv_np)
    list_train_score.append(score_tr)
    list_cv_score.append(score_cv)
    return model


# 模型融合的bagging
'''bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)
test = test_df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId': test_data['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
result.to_csv("C:/python/machine learning/kaggle titanic/logistic_regression_bagging_predictions.csv", index=False)'''


# joblib.dump(clf,'clf.dat')

if __name__ == '__main__' :
    #数据读取和处理
    all_data = pd.read_csv("train.csv")
    all_data.info()

    all_data, rfr = set_missing_ages(all_data)
    all_data = set_cabin_type(all_data)
    train_data, cv_data = model_selection.train_test_split(all_data, test_size=0.3, random_state=0)

    all_df, scaler, age_scale, fare_scale = data_processing(all_data)
    train_split, cv_split = model_selection.train_test_split(all_df, test_size=0.3, random_state=0)

    train_np, cv_np = create_train_cv_set(train_split, cv_split)


    # 建立模型
    list_train_score = []
    list_cv_score = []

    #逻辑回归
    model_logistic = linear_model.LogisticRegression(C=1.0, tol=1e-6, penalty='l2')
    model_logistic = list_score(list_train_score,list_cv_score,model_logistic,train_np,cv_np)

    #决策树
    model_decisiontree = DecisionTreeClassifier(max_depth=3,random_state = 0)
    model_decisiontree = list_score(list_train_score,list_cv_score,model_decisiontree,train_np,cv_np)

    #随机森林
    model_rfr = RandomForestClassifier(max_depth=3,n_estimators = 100)
    model_rfr = list_score(list_train_score,list_cv_score,model_rfr,train_np,cv_np)

    #支持向量机
    model_svc = SVC()
    model_svc = list_score(list_train_score,list_cv_score,model_svc,train_np,cv_np)

    #高斯分布下的朴素贝叶斯
    model_bayes = GaussianNB(var_smoothing = 1e-6)
    model_bayes = list_score(list_train_score,list_cv_score,model_bayes,train_np,cv_np)

    #knn
    model_knn = KNeighborsClassifier(n_neighbors = 3)
    model_knn = list_score(list_train_score,list_cv_score,model_knn,train_np,cv_np)

    #梯度提升树
    model_GBDT = GradientBoostingClassifier(n_estimators = 100,max_depth=3)
    model_GBDT = list_score(list_train_score,list_cv_score,model_GBDT,train_np,cv_np)



    '''#kmeans
    model_kmeans = KMeans(n_clusters = 2)
    train_y = train_np[:, 0]
    train_X = train_np[:,1:]
    cv_y = cv_np[:,0]
    cv_X = cv_np[:,1:]
    model_kmeans.fit(train_X,train_y)

    from sklearn import metrics
    y1 = model_kmeans.predict(train_X)
    y2 = model_kmeans.predict(cv_X)
    score_kmeans_tr = metrics.silhouette_score(train_X, y1)
    score_kmeans_cv = metrics.silhouette_score(cv_X, y2)
    list_train_score.append(score_kmeans_tr)
    list_cv_score.append(score_kmeans_cv)'''

    #xgboost
    model_xgb = xgb.XGBClassifier(max_depth = 2,n_estimators = 50, random_state = 0)
    model_xgb = list_score(list_train_score,list_cv_score,model_xgb,train_np,cv_np)

    # adaboost
    model_ada = AdaBoostClassifier(n_estimators=100)
    model_ada = list_score(list_train_score, list_cv_score, model_ada, train_np, cv_np)

    #MLP神经网络
    model_mlp = MLPClassifier(alpha=0.1)
    model_mlp = list_score(list_train_score,list_cv_score,model_mlp,train_np,cv_np)

    print(list_train_score)
    print(list_cv_score)
    d1 = pd.DataFrame([list_train_score,list_cv_score])
    d1.to_csv('accuracy.csv')







