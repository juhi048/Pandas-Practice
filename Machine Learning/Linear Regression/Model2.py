import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r"C:\Users\user\OneDrive\Documents\Pandas Projects\Machine Learning\Linear Regression\Iris.csv")

print(data)
print(data.columns)
lr = LinearRegression()
x=data[["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]]
y=data[["PetalLengthCm"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

print(y_pred[0:5])
print(y_test[0:5])

print(mean_squared_error(y_test, y_pred))



