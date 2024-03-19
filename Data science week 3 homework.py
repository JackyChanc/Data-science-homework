import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Brooklyn_Bridge_Automated_Pedestrian_Counts_Demonstration_Project')
#Question 1
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
weekdays_data = df[df['day_of_week'].isin([0, 1, 2, 3, 4])]

# Step 4: Plot Line Graph
plt.figure(figsize=(10, 6))
plt.plot(df['day_of_week'],df['Pedestrian'],color = "blue")
weekdays_data.groupby('day_of_week')['pedestrian_count'].mean().plot(kind='line', marker='o')
plt.xticks(range(5), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
plt.xlabel('Day of the Week')
plt.ylabel('Pedestrian Count')
plt.title('Pedestrian Counts on Brooklyn Bridge (Weekdays)')
plt.gird(True)
plt.show()
