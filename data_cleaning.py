#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 19:34:55 2023

@author: prateekbangwal
"""
import pandas as pd

df = pd.read_csv('glassdoor_salary.csv')

#Salary parsing


#removing bad salary estimates

df = df[df['Salary Estimate'] != '-1']

#get hourly information
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 
                                           if 'per hour' in x.lower() else 0)
#employer provided
df['Employer Provided'] = df['Salary Estimate'].apply(lambda x: 1
                                                      if 'employer provided salary:'
                                                      in x.lower() else 0)

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])  
remove_kd = salary.apply(lambda x:x.replace('K','').replace('$',''))

remove_hr_employer_ps = remove_kd.apply(lambda x: x.lower().replace('per hour','').
                                        replace('employer provided salary:',''))

#split the salary

df['min_salary'] = remove_hr_employer_ps.apply(lambda x:int(x.split('-')[0]))
df['max_salary'] = remove_hr_employer_ps.apply(lambda x:int(x.split('-')[1]))

df['avg_salary'] = (df['max_salary'] - df['min_salary'])/2


#Company Name text only

df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating']<0
                             else x['Company Name'][:-3], axis = 1)

#State field

df['job_state'] = df['Location'].apply(lambda x:x.split(',')[1])
print(df['job_state'].value_counts())

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)


#Company Age


df['age_company'] = df['Founded'].apply(lambda x : x if x<1 else 2023 - x)

#parsing of job description python, ML etc

df['python_jb'] = df['Job Description'].apply(lambda x:1 if 'python' in x.lower() else 0)

df['python_jb'].value_counts()

df['spark'] = df['Job Description'].apply(lambda x:1 if 'spark' in x.lower() else 0)
df['spark'].value_counts()


df['excel'] = df['Job Description'].apply(lambda x:1 if 'excel' in x.lower() else 0)
df['excel'].value_counts()


df['r_lang'] = df['Job Description'].apply(lambda x:1 if 'R language' in x.lower() or 
                                      'r-studio' in x.lower() or 
                                      'r studio' in x.lower() else 0)
df['r_lang'].value_counts()

df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['aws'].value_counts()

df.columns
df_out = df.drop(['Unnamed: 0'], axis = 1)

df_out.to_csv('salary_data_cleaned.csv', index=False)

