import numpy as np
import pandas as pd
def gradient_descent(gradient,start,learn_rate,n_iter):
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate * gradient(vector)
    vector += diff
  return vector
def gradient_descent(gradient,start,learn_rate,n_iter=50,tolerance = 1e-06):
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate*gradient(vector)
    if np.all(np.abs(diff)<=tolerance):
      break
    vector += diff
  return vector
 
 print(gradient_descent(gradient=lambda v:2*v,start=10.0,learn_rate=0.2))
 print(gradient_descent(gradient=lambda v:2*v,start=10.0,learn_rate=0.8))
 print(gradient_descent(gradient=lambda v:2*v,start=10.0,learn_rate=0.005))
 print(gradient_descent(gradient=lambda v:2*v,start=10.0,learn_rate=0.005,n_iter=100))
 print(gradient_descent(gradient=lambda v:2*v,start=10.0,learn_rate=0.005,n_iter=1000))
 print(gradient_descent(gradient=lambda v:2*v,start=10.0,learn_rate=0.005,n_iter=2000))
 print(gradient_descent(gradient=lambda v:4*v**3 - 10*v -3,start=0,learn_rate=0.2))
 print(gradient_descent(gradient=lambda v:4*v**3 - 10*v -3,start=0,learn_rate=0.1))
 
