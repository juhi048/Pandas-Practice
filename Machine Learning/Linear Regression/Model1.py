import pandas as pd

df = pd.read_csv(r"C:\Users\user\OneDrive\Documents\Pandas Projects\Machine Learning\Linear Regression\Iris.csv")

# print(len(df.head()))
# print(df[df.Species.str.contains("Iris-setosa")])
# print(len(df.SepalLengthCm > 3))
# print(df.iloc())
# print(df.columns)

from matplotlib import pyplot as plt
import seaborn as sns

# let's say want to find relation between sepal length and petal length
# sns.scatterplot(x="PetalLengthCm", y="SepalLengthCm", data=df, hue="Species")
# plt.show()

# Now, let's say we want to see how sepal length is changing wrt sepal width
#     y = mx+c   y -> dependent     x = independent

y = df[["SepalLengthCm"]]
x = df[["SepalWidthCm"]]

# For implementing any of the machine learning models like linear regression, Random forest etc we use the libary sklearn i.e scikit learn 
# so , we will divide our data into training and testing set with the help of the scikit learn

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)    # data is divided into 30:70   30 is the testing set

# x_train, x_test, y_train, y_test  --> this sequence is important 
# x_train -> training set of indepedent data
# x_test -> testing set of independent data
# y_test -> testing set of dependent data
# y_train -> training set of dependent data   

# for implement linear regression in our model we'll import linear regression 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# print(lr.fit(x_train, y_train))
lr.fit(x_train, y_train)

# print(lr.predict(x_test)) 
y_pred = lr.predict(x_test)
print(y_test[0:5])
print(y_pred[0:5])

# Found testing and predicting value of y 

# Now we'll calculate error in prediction value 

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))

# Output : 0.6078347570872679

# Note :  Value of mean_squared_error should be greater or equal to zero 
#  lesser the value of mean_squared_error , it is good i.e value should be less only
