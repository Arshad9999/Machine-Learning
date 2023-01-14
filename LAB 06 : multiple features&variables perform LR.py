import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
house_data = pd.read_csv('kc_house_data.csv')
house_data.columns
house_data.isna().sum()
house_data.dropna(axis=0,inplace=True)
house_data.isna().sum()
sns.heatmap(house_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
house_data.describe()
house_data.hist(figsize=(30,20))
plt.show()
corr = house_data.corr()
sns.heatmap(corr)
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(house_data.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=False);
sns.distplot(house_data['price'])
house_data.columns
feature_cols = ['sqft_living','bedrooms','bathrooms','floors','sqft_above','sqft_basement','lat','long']
x = house_data[feature_cols]#predictor
y = house_data.price#response
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_poly=scaler.transform(x)
print(x_poly)
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2)
lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.intercept_)
print(lr.coef_)
test_data_prediction = lr.predict(x_test)
mse = mean_squared_error(np.array(y_test).reshape(-1,1),test_data_prediction)
mse
lr.score(x_train,y_train)*100
