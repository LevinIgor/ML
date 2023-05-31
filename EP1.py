# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
dataframe = pd.read_csv('data.csv')

# Setting up our predictor and target variables
predictor = dataframe[['rating']]
target = dataframe['salary']

# Splitting the data into training and testing sets
predictor_train, predictor_test, target_train, target_test = train_test_split(predictor, target, test_size=0.3, random_state=100)

# Fitting a linear regression model
linear_model = LinearRegression()
linear_model.fit(predictor_train, target_train)

# Predicting salary using the fitted model
target_predicted = linear_model.predict(predictor_test)

# Calculating the Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((target_test - target_predicted) / target_test)) * 100

# Printing the coefficients and MAPE
print(f"Intercept: {linear_model.intercept_:.5f}")
print(f"Coefficient: {linear_model.coef_[0]:.5f}")
print(f"MAPE: {mape:.5f}")

# Scatterplot
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataframe, x='rating', y='salary', color='dodgerblue', label='Actual values')
sns.lineplot(x=predictor_test['rating'], y=target_predicted, color='red', label='Predicted values')
plt.title('Linear Regression', fontsize=14)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Salary', fontsize=12)
plt.legend(loc='upper left')
plt.show()
