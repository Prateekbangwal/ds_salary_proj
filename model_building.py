#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:46:47 2023

@author: prateekbangwal
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data_eda.csv')

#choose relevant columns

df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership',
               'Industry', 'Sector','Revenue','hourly', 
             'Employer Provided','no_of_Competitors','same_state','job_state', 
             'age_company', 'python_jb','spark', 'excel',
             'aws', 'job_simp', 'seniority','desc_length']]
#get dummy data

df_dum = pd.get_dummies(df_model)

#train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary',axis = 1)
y = df_dum.avg_salary.values

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size =0.2, 
                                                    random_state = 42)

#multiple linear regression

#using statsmodel
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)

model = sm.OLS(y,X_sm)

model.fit().summary()

#using sklearn linear model

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(x_train, y_train)

np.mean(cross_val_score(lm,x_train,y_train, scoring = 'neg_mean_absolute_error', cv = 3))


#lasso regression

lm_l = Lasso()

lm_l.fit(x_train, y_train)

np.mean(cross_val_score(lm_l,x_train,y_train, scoring = 'neg_mean_absolute_error', cv = 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/200)
    lml = Lasso(alpha=i/200)
    error.append(np.mean(cross_val_score(lml,x_train,y_train, 
                                         scoring = 'neg_mean_absolute_error',
                                         cv = 3)))
    
plt.plot(alpha, error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

#random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

np.mean(cross_val_score(rf, x_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))


#tune models using GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':range(10,300,10),
              'criterion':('mse','mae'),
              'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv = 3)
gs.fit(x_train, y_train)

gs.best_score_
gs.best_estimator_


#test ensembles
tpred_lm = lm.predict(x_test)
tpred_lm_l = lm_l.predict(x_test)
tpred_rf = gs.predict(x_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lm_l)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

import pickle
pickl = {'model':gs.best_estimator_}
pickle.dump(pickl, open('model_file'+".p","wb"))

file_name = "model_file.p"

with open(file_name,'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    
model.predict(np.array(list(x_test.iloc[1,:])).reshape(1,-1))[0]

list(x_test.iloc[1,:])

