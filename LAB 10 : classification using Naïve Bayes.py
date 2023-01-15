import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
df = pd.read_csv('games1.csv',encoding='utf-8')
df.iloc[:,:12]
df['winner'].value_counts()
df['rating_difference']=df['white_rating']-df['black_rating']
df['white_win']=df['winner'].apply(lambda x:1 if x=='white' else 0)
df['match_outcome']=df['winner'].apply(lambda x:1 if x == 'white' else 0 if x=='draw' else -1)
df.iloc[:,13:]

def mfunc(x,y,typ):
  X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)
  model = typ
  clf = model.fit(X_train,y_train)
  pred_labels = model.predict(X_test)
  print('Classes: ', clf.classes_)
  if str(typ)=='GaussianNB()':
    print('Class Priors: ',clf.class_prior_)
  else:
    print('Class Log Priors: ',clf.class_log_prior_)
  print('--------------------------------------------------------')
  score = model.score(X_test, y_test)
  print('Accuracy Score: ', score)
  print('--------------------------------------------------------')
  print(classification_report(y_test, pred_labels))
  return X_train, X_test, y_train, y_test, clf, pred_labels
  
X=df[['rating_difference', 'turns']]
y=df['white_win'].values
X_train, X_test, y_train, y_test, clf, pred_labels, = mfunc(X, y, GaussianNB())
mesh_size = 5
margin = 1
x_min, x_max = X.iloc[:,0].fillna(X.mean()).min() - margin, X.iloc[:,0].fillna(X.mean()).max()+margin
y_min, y_max = X.iloc[:,1].fillna(X.mean()).min() - margin, X.iloc[:,1].fillna(X.mean()).max()+margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
trace_specs = [[X_test, y_test, 0, 'Test', 'red'], [X_test, y_test, 1, 'Test', 'blue']]
fig = go.Figure(data=[go.Scatter(x=X[y==label].iloc[:, 0], y=X[y==label].iloc[:, 1], name=f'{split} data, Actual Class: {label}', mode='markers', marker_color=marker) for X, y, label, split, marker in trace_specs])

fig.update_traces(marker_size=2, marker_line_width=0)
fig.update_xaxes(range=[-1600, 1500])
fig.update_yaxes(range=[0,345])
fig.update_layout(title_text="Decision Boundary for Naive Bayes Model", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.add_trace(go.Contour(x=xrange, y=yrange, z=Z, showscale=True, colorscale='magma', opacity=1, name='Score', hoverinfo='skip'))
fig.show()
