import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima = pd.read_csv('pima-indians-diabetes.csv',header=None,names=col_names)
plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(7,7))
plt.pie(x=pima['label'].value_counts(), colors=["firebrick","seagreen"], labels=["Diabetes Patients","Healthy Patients"], shadow = True, explode = (0, 0.1))
plt.show()
plt.bar([1,0],pima.label.value_counts() , color ='maroon', width = 0.4)
plt.xlabel("Diabetes Disease")
plt.ylabel("Count of the Diabetes Disease")
plt.title("Count of the Diabetes Disease in dataset")
plt.show()
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(pima.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
x = pima.drop('label',axis=1) #features
y = pima.label #target variable
pima['label'].value_counts()
from imblearn.combine import SMOTEENN
sn = SMOTEENN(random_state=0)
sn.fit(x,y)
x,y=sn.fit_resample(x,y)
y.value_counts()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
x_test_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score
test_data_accuracy = accuracy_score(x_test_pred, y_test)
print('Accuracy on Test data using Logistice Regression : ', test_data_accuracy*100)
input_data = (1,89,66,23,94,28.1,0.167,21)
input_data_as_numpy_array= np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = lr.predict(input_data_reshaped)

if (prediction[0]== 0):
  print('According to the given details person does not have a diabetes Disease')
 
else:
  print('According to the given details person has diabetes Disease')
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,x_test_pred)
print("Accuracy:",metrics.accuracy_score(y_test,x_test_pred))
print("Precision:",metrics.precision_score(y_test,x_test_pred))
print("Recall:",metrics.recall_score(y_test,x_test_pred))
y_pred_proba = lr.predict_proba(x_test)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_proba)
auc = metrics.roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label='data 1, auc ='+str(auc))
plt.legend(loc=4)
plt.show()
