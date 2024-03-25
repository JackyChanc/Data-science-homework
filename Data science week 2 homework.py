#Question 1
import numpy as np
import pandas as pd
import random
df1 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
filtered_df = df1.loc[::20, ['Manufacturer', 'Model', 'Type']]
#Question 2

df1['Min.Price'].fillna(df1['Min.Price'].mean(), inplace=True)
df1['Max.Price'].fillna(df1['Max.Price'].mean(), inplace=True)
print(df1[['Min.Price', 'Max.Price']])


#Question 3
df2 = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
row_sum = df2[df2.sum(axis = 1) > 100]

#Question 4
x = np.random.random_integers((4,4))