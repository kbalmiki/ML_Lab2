import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = 'inc_utf.csv'
data = pd.read_csv(file_path)

# Explicitly handle the '100+' entry
data['age'] = data['age'].apply(lambda x: x.replace(' years', '').replace('100+', '100')).astype(int)

grouped_by_age_mean = data.groupby('age')['2020'].mean().reset_index()

# Extract the age and income columns
age = grouped_by_age_mean['age'].values.reshape(-1, 1)
income = grouped_by_age_mean['2020'].values

# Split the data into training and validation sets (80% training, 20% validation)
age_train, age_val, income_train, income_val = train_test_split(age, income, test_size=0.2, random_state=42)

# Sort the training and validation data based on age
sorted_idx_train = np.argsort(age_train.flatten())
sorted_idx_val = np.argsort(age_val.flatten())

age_train = age_train[sorted_idx_train]
income_train = income_train[sorted_idx_train]
age_val = age_val[sorted_idx_val]
income_val = income_val[sorted_idx_val]

# Function to plot polynomial regression for different degrees
def plot_polynomial_regression(age_train, income_train, age_val, income_val, max_degree=10):
    plt.figure(figsize=(10, 6))

    # Scatter plot of training data points
    plt.scatter(age_train, income_train, color='blue', label='training data points')

    # Scatter plot of validation data points
    plt.scatter(age_val, income_val, color='green', label='validation data points', marker='x')

    mse_train_list = []
    mse_val_list = []

    # Try polynomial degrees from 1 to max_degree
    for degree in [2, 3, 5, 8]:
        poly_features = PolynomialFeatures(degree=degree)

        # Transform both training and validation data
        age_train_poly = poly_features.fit_transform(age_train)
        age_val_poly = poly_features.transform(age_val)

        # Fit Linear Regression model on training data
        model = LinearRegression()
        model.fit(age_train_poly, income_train)

        # Predict for both training and validation sets
        income_train_pred = model.predict(age_train_poly)
        income_val_pred = model.predict(age_val_poly)

        # Calculate MSE for training and validation sets
        mse_train = mean_squared_error(income_train, income_train_pred)
        mse_val = mean_squared_error(income_val, income_val_pred)

        mse_train_list.append(mse_train)
        mse_val_list.append(mse_val)

        # Plot the polynomial regression line for data (sorted)
        plt.plot(age_val, income_val_pred,
                 label=f'degree = {degree}, Train MSE = {mse_train:.2f}, Val MSE = {mse_val:.2f}', linestyle='-')

    # Add plot labels and title
    plt.xlabel('Age')
    plt.ylabel('Average Income')
    plt.title('Polynomial Regression with Training and Validation Data')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the polynomial regression
plot_polynomial_regression(age_train, income_train, age_val, income_val)
