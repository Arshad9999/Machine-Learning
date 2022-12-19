
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt %matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
house_data = pd.read_csv('kc_house_data.csv')
house_data.head()
house_data.isna().sum()
house_data.describe()
house_data.hist(figsize=(30,20))
plt.show()
corr = house_data.corr()
sns.heatmap(corr)
feature_cols = 'sqft_living'
x = house_data[feature_cols]#predictor
y = house_data.price#response
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lr = LinearRegression()
lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
print(lr.intercept_)
print(lr.coef_)
price = lr.intercept_[0] + lr.coef_[0][0]*1000
print(price)
lr.predict(np.array(1000).reshape(-1,1))
mse = mean_squared_error(np.array(y_test).reshape(-1,1),lr.predict(np.array(x_test).reshape(- 1,1)))
np.sqrt(mse)
lr.score(np.array(x_test).reshape(-1,1),np.array(y_test).reshape(-1,1))

