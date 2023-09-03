import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df=pd.read_csv("C:\\Users\\HP\\Downloads\\archive (2)\\Real estate.csv")
df.head()
df.info()

sns.pairplot(df)
plt.show()

sns.distplot(df['Y house price of unit area'], color='green')
plt.show()

x=df.drop(['Y house price of unit area', 'No'], axis=1)
y=df['Y house price of unit area']
x.head(10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
x_train.head(10)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

pd.DataFrame(reg.coef_, x.columns, columns=['coefficient'])

reg.intercept_

df['Y house price of unit area'].value_counts()

y_predict = reg.predict(x_test)
pd.DataFrame({'Test':y_test, 'Prediction':y_predict}).head(10)

from sklearn import metrics
MAE = metrics.mean_absolute_error(y_predict, y_test)
MSE = metrics.mean_squared_error(y_predict, y_test)
RMSE = np.sqrt(MSE)

pd.DataFrame([MAE, MSE, RMSE], index=['MAE', 'MSE', 'RMSE'], columns=['Metrics'])

df['Y house price of unit area'].mean()

residuals = y_test-y_predict
sns.scatterplot(x=y_test, y=y_predict, color='olive')
plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.show()

sns.distplot(residuals, color='r', hist=False)

sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', ls='--')

plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color = "green", s = 35, edgecolor='black', label = 'Train data')
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color = "red", s = 35, edgecolor='black', label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, colors='#5e03fc', linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()

import pickle

# Train your linear regression model (code omitted for brevity)
reg = LinearRegression()
reg.fit(x_train, y_train)

# Save the trained model as a pickle file
with open('real_estate_model.pkl', 'wb') as file:
    pickle.dump(reg, file)
    
import os

pickle_file_path = os.path.abspath('linear_regression_model.pkl')
print(f"Pickle file path: {pickle_file_path}")


