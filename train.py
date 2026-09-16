import numpy as np 
from sklearn.linear_model import LinearRegression
import joblib

X = np.array([[1],[2], [3], [4], [5]])
y = np.array([[10],[20],[30],[40],[50]])

model = LinearRegression()
model.fit(X,y)

model.predict([[6]])

joblib.dump(model,"model.pkl")