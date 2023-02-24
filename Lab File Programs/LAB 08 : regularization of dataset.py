import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
data = pd.read_csv('kc_house_data.csv')
data.isnull().sum()
data.dropna(inplace=True)
dropColumns = ['id','date','zipcode']
data = data.drop(dropColumns, axis=1)
y=data['price']
x = data.drop('price',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
lr = LinearRegression()
lr.fit(x_train,y_train)
cross_val_scores_ridge = []
alpha=[]

for i in range(1,9):
  ridgeModel = Ridge(alpha = i*0.25)
  ridgeModel.fit(x_train,y_train)
  scores = cross_val_score(ridgeModel,x,y,cv=10)
  avg_cross_val_score = mean(scores)*100
  cross_val_scores_ridge.append(avg_cross_val_score)
  alpha.append(i*0.25)
for i in range(0,len(alpha)):
  print(str(alpha[i])+':'+str(cross_val_scores_ridge[i]))
cross_val_scores_lasso=[]
Lambda = []

for i in range(1,9):
  lassoModel = Lasso(alpha = i*0.25,tol=0.0925)
  lassoModel.fit(x_train,y_train)
  scores = cross_val_score(lassoModel,x,y,cv=10)
  avg_cross_val_score = mean(scores)*100
  cross_val_scores_lasso.append(avg_cross_val_score)
  Lambda.append(i*0.25)
for i in range(0,len(alpha)):
  print(str(alpha[i])+':'+str(cross_val_scores_lasso[i]))
lassoModelChosen = Lasso(alpha=2,tol=0.0925)
lassoModelChosen.fit(x_train,y_train)
print(lassoModelChosen.score(x_test,y_test))
models = ['Linear Regression','Ridge Regression','Lasso Regression']
scores=[lr.score(x_test,y_test),ridgeModel.score(x_test,y_test), lassoModelChosen.score(x_test,y_test)]
mapping = {}
mapping['Linear Regression'] = lr.score(x_test,y_test)
mapping['Ridge Regression'] = ridgeModel.score(x_test,y_test)
mapping['Lasso Regression']=lassoModelChosen.score(x_test,y_test)
for key,val in mapping.items():
  print(str(key)+':'+str(val))
plt.bar(models,scores)
plt.xlabel('Regression Models')
plt.ylabel('Score')
plt.show()
