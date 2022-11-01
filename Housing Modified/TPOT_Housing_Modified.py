import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

os.chdir("C:/Users/DELL/Desktop/UNI/Sem 5 2022/ZA3302-Machine Learning/Assignments/AutoML Assignment/Housing Modified")

df = pd.read_csv('Housing_Modified.csv')

df.head()

df['stories'] = df['stories'].map({'one': 0, 'two': 1, 'three': 2, 'four': 3})
df['driveway'] = df['driveway'].map({'yes': 1, 'no': 0})
df['recroom'] = df['recroom'].map({'yes': 1, 'no': 0})
df['fullbase'] = df['fullbase'].map({'yes': 1, 'no': 0})
df['gashw'] = df['gashw'].map({'yes': 1, 'no': 0})
df['airco'] = df['airco'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})

df.info()

# checking for missing
df.isnull().sum()


x = df.iloc[:,1:] # input
y = df.iloc[:,0] # output - price

from sklearn.model_selection import cross_val_score

# Individual Algorithm - find the mean of the algorithm
cv_scores = cross_val_score(LogisticRegression(), x, y, cv=10)
print(cv_scores)
print(np.mean(cv_scores))

rf_cv_scores = cross_val_score(RandomForestClassifier(), x, y, cv=10)
print(rf_cv_scores)
print(np.mean(rf_cv_scores))

rf_cv_scores2 = cross_val_score(RandomForestClassifier(n_estimators=100,max_depth=2),x,y,cv=10)
print(rf_cv_scores2)
print(np.mean(rf_cv_scores2))

import tpot
print(dir(tpot)) #to check the methods and attributes available at TPOT

# Split in train and test
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Init
# from tpot import TPOTClassifier
# tpot = TPOTClassifier(generations=5,verbosity=2)

# # Fit data
# tpot.fit(x_train,y_train)

# # Export the result
# tpot.export('tpot_ml_pipeline.py')

# Init
from tpot import TPOTRegressor
tpot = TPOTRegressor(generations=5,verbosity=2)

# Fit data
tpot.fit(x_train,y_train)

# Export the result
tpot.export('tpot_ml_pipeline2.py')
