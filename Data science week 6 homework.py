import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 101)

url = "https://raw.githubusercontent.com/abirami1998/NYU-Data-Science-Bootcamp-Spring-2024/main/Week%206/employee.csv"
df = pd.read_csv(url)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(df_test.drop(columns=["target_column"]))
#1
predictions = model.predict(X_test_scaled)
#2
mae = mean_absolute_error(df_test["target_column"], predictions)
mse = mean_squared_error(df_test["target_column"], predictions)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)