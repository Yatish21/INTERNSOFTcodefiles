# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#IMPORTING LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt

#READING THE DATA FROM YOUR FILES

data = pd.read_csv('advertising.csv')
data.head()

#TO VISUALIZE DATA

fig, axs = plt.subplots(1,3,sharey = True)
data.plot(kind = 'scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind = 'scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind = 'scatter',x='Newspaper',y='Sales',ax=axs[2])

#CREATED X&Y FOR LINEAR REGRESSION
feature_cols = ['TV']
X= data[feature_cols]
y = data.Sales

#IMPORTING LINEAR REGRESSION ALGORITHM FOR SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)


# Y = a+bx
# Y = result a = 6.974821488229891 b = 0.05546477 x = 50

result = 6.97 +0.0554*50
print(result)


#CREATE A DATAFRAME WITH MIN AND MAX VALUE OF THE TABLE
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds = lr.predict(X_new)
preds

data.plot(kind = 'scatter',x = 'TV',y = 'Sales')
plt.plot(X_new,preds,color = 'g',linewidth = 2)

import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int() 

#FINDING THE PROBABILITY VALUES
lm.pvalues


#FINDING THE R-SQUARED VALUES
lm.rsquared



#CREATING MULTI LINEAR REGRESSION
feature_cols = ['TV','Radio','Newspaper']
X= data[feature_cols]
y = data.Sales


lr = LinearRegression()
lr.fit(X,y)



print(lr.intercept_)
print(lr.coef_)



lm = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()



lm = smf.ols(formula = 'Sales ~ TV+Radio',data=data).fit()
lm.conf_int()
lm.summary()



























