import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
data = pd.read_csv('data.csv')

# Check which variables are categorical
categorical_variables = data.select_dtypes(include=['object']).columns

# Encoding categorical variables if present
if len(categorical_variables) > 0:
    data_encoded = pd.get_dummies(data, columns=categorical_variables)
else:
    data_encoded = data

# Ensure that the target variable is present in the dataset
target_variable = 'salary'
assert target_variable in data_encoded.columns, f"Target variable '{target_variable}' is not in the dataset"

# Splitting into X and y
X = data_encoded.drop(target_variable, axis=1)  # Predictors
y = data_encoded[target_variable]  # Target variable

# Step 3: Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and calculation of MAPE
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Printing coefficients and MAPE
coef_b0 = model.intercept_
coef_b1 = model.coef_

for i, b in enumerate(coef_b1):
    print(f"Coefficient b{i+1}: {b:.5f}")
print(f"Intercept b0: {coef_b0:.5f}")
print(f"MAPE: {mape:.5f}")

# Setting the plot style
sns.set_style("whitegrid")

# Building scatter plots between each predictor and the target variable
for column in X.columns[:5]:
    plt.figure(figsize=(10, 6))

    # Fitting a linear regression model for each predictor
    model_single = LinearRegression()
    model_single.fit(X[[column]], y)

    # Predicting
    y_pred_single = model_single.predict(X[[column]])

    sns.scatterplot(x=X[column], y=y, color='dodgerblue', label='Actual values')
    sns.lineplot(x=X[column], y=y_pred_single, color='red', label='Predicted values')
    plt.title(f'Linear Regression for {column}', fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.legend(loc='upper left')
    plt.show()
