
Airbnb.ipynb
Airbnb.ipynb_Notebook unstarred
[2]
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import learning_curve
# set up notebook to show all outputs in a cell, not only last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np
First Goal:

-derive actionable insights for hosts and potential renters

-predict rental prices based on relevant features

[16]
0s
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
[ ]
!pip install kaggle
import os
Requirement already satisfied: kaggle in /usr/local/lib/python3.10/dist-packages (1.6.12)
Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.10/dist-packages (from kaggle) (1.16.0)
Requirement already satisfied: certifi>=2023.7.22 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2024.2.2)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.8.2)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.31.0)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from kaggle) (4.66.2)
Requirement already satisfied: python-slugify in /usr/local/lib/python3.10/dist-packages (from kaggle) (8.0.4)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.0.7)
Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from kaggle) (6.1.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->kaggle) (0.5.1)
Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.10/dist-packages (from python-slugify->kaggle) (1.3)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.7)
[14]
url = '/content/drive/MyDrive/kaggle/AB_NYC_2019.csv'
df = pd.read_csv(url)
Inspecting data
[ ]
df.head()

[ ]
df.shape
(48895, 16)
[ ]
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 48895 entries, 0 to 48894
Data columns (total 16 columns):
 #   Column                          Non-Null Count  Dtype
---  ------                          --------------  -----
 0   id                              48895 non-null  int64
 1   name                            48879 non-null  object
 2   host_id                         48895 non-null  int64
 3   host_name                       48874 non-null  object
 4   neighbourhood_group             48895 non-null  object
 5   neighbourhood                   48895 non-null  object
 6   latitude                        48895 non-null  float64
 7   longitude                       48895 non-null  float64
 8   room_type                       48895 non-null  object
 9   price                           48895 non-null  int64
 10  minimum_nights                  48895 non-null  int64
 11  number_of_reviews               48895 non-null  int64
 12  last_review                     38843 non-null  object
 13  reviews_per_month               38843 non-null  float64
 14  calculated_host_listings_count  48895 non-null  int64
 15  availability_365                48895 non-null  int64
dtypes: float64(3), int64(7), object(6)
memory usage: 6.0+ MB
[ ]
df['last_review']=pd.to_datetime(df['last_review'])
df.info()
df['last_review'].head(5)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 48895 entries, 0 to 48894
Data columns (total 16 columns):
 #   Column                          Non-Null Count  Dtype
---  ------                          --------------  -----
 0   id                              48895 non-null  int64
 1   name                            48879 non-null  object
 2   host_id                         48895 non-null  int64
 3   host_name                       48874 non-null  object
 4   neighbourhood_group             48895 non-null  object
 5   neighbourhood                   48895 non-null  object
 6   latitude                        48895 non-null  float64
 7   longitude                       48895 non-null  float64
 8   room_type                       48895 non-null  object
 9   price                           48895 non-null  int64
 10  minimum_nights                  48895 non-null  int64
 11  number_of_reviews               48895 non-null  int64
 12  last_review                     38843 non-null  datetime64[ns]
 13  reviews_per_month               38843 non-null  float64
 14  calculated_host_listings_count  48895 non-null  int64
 15  availability_365                48895 non-null  int64
dtypes: datetime64[ns](1), float64(3), int64(7), object(5)
memory usage: 6.0+ MB
0   2018-10-19
1   2019-05-21
2          NaT
3   2019-07-05
4   2018-11-19
Name: last_review, dtype: datetime64[ns]
We'll now inspect (and possibly clean/filter) the data

Things we need to do include:

Missing Values: We need to check if there are any missing values in the data. Sometimes, certain rows might have missing information, which can be represented as 'None', 'NaN', or even '0' or '-1'. It's essential to distinguish between actual missing values and valid values like '0' or '-1'.

Numeric Fields: For numerical fields, we'll look at the minimum and maximum values of each field. We'll also check if the median falls within our expected range.

Non-Numeric Fields: For non-numeric fields, we'll check the number of unique values in each field and ensure they match our expectations. We'll also examine the consistency of factor levels throughout the data.

Variable Relationships: We'll assess if the relationships between variables align with our expectations. This can involve visual evaluation and examining summary statistics.

Time Series Data: If the data is a time series, we'll analyze the trend of each variable over time and ensure it aligns with our expectations.

[ ]
df.isnull().sum()
id                                    0
name                                 16
host_id                               0
host_name                            21
neighbourhood_group                   0
neighbourhood                         0
latitude                              0
longitude                             0
room_type                             0
price                                 0
minimum_nights                        0
number_of_reviews                     0
last_review                       10052
reviews_per_month                 10052
calculated_host_listings_count        0
availability_365                      0
dtype: int64
[ ]
df[(df['number_of_reviews'] == 0) & (df['last_review'].notnull())]

Also, we checked that all our None values are explicit... no use of -1 or 0 or anything. Around 20% of our last review and reviews per month are null. This is a big number. But all these null values are because there's 0 reviews, as evident from above block of code.

Now we'll check the range of values of our numeric fields:

[ ]
min_last_review= df.last_review.min()
max_last_review = df.last_review.max()
min_last_review, max_last_review
(Timestamp('2011-03-28 00:00:00'), Timestamp('2019-07-08 00:00:00'))
[ ]
min= df.latitude.min()
max = df.latitude.max()
min, max
(40.49979, 40.91306)
[ ]
min= df.longitude.min()
max = df.longitude.max()
min, max
(-74.24442, -73.71299)
[ ]
min_price= df.price.min()
max_price = df.price.max()
min_price, max_price
(0, 10000)
[ ]
min= df.minimum_nights.min()
max = df.minimum_nights.max()
min, max
(1, 1250)
[ ]
min= df.reviews_per_month.min()
max = df.reviews_per_month.max()
min, max
(0.01, 58.5)
[ ]
min= df.availability_365.min()
max = df.availability_365.max()
min, max
(0, 365)
[ ]
min= df.number_of_reviews.min()
max = df.number_of_reviews.max()
min, max
(0, 629)
All the ranges fall in the expected range.

[ ]
median_price = df['price'].median()
median_price
106.0
median seems to fall within our expected range too

We'll now look at our non-numeric fields.

[ ]
df.neighbourhood_group.value_counts()
#df.neighbourhood.value_counts()
print()
df.room_type.value_counts()
neighbourhood_group
Manhattan        21661
Brooklyn         20104
Queens            5666
Bronx             1091
Staten Island      373
Name: count, dtype: int64

room_type
Entire home/apt    25409
Private room       22326
Shared room         1160
Name: count, dtype: int64
So, the consistency factor of the data seems pretty high

Let us now see if we can work to find variable relationships using numpy and pandas

relevant fields:

neighbourhood_group
neighbourhood
latitude
longitude
room_type
number_of_reviews or reviews_per_month or last_review
calculated_host_listings_count
availability_365
minimum_nights
price
[ ]
df.groupby('neighbourhood_group')['price'].describe()

So, it seems like prices in Manhattan and Brooklyn seem to be higher, while they are the lowest at Bronx.

[ ]
df.groupby('room_type')['price'].describe()

It seems like the cost of entire home/apt are the highest, then private room, then shared rooms (as expected).

[ ]
df.groupby('number_of_reviews')['price'].describe()

[ ]
from matplotlib import pyplot as plt
_df_23['mean'].plot(kind='line', figsize=(8, 4), title='mean')
plt.gca().spines[['top', 'right']].set_visible(False)

So, it seems like on the overall, as the number of reviews increase, the price decreases. The trend does have a lot of variations, but overall, that seems to be the case. The trend seems to be clearer if we look more closely to start of the graph, since the value_counts for number_of_reviews decreases pretty rapidly to be mostly single digit numbers after number of reviews=150 or so.

[ ]
df.groupby('availability_365')['price'].describe()

#df.groupby('calculated_host_listings_count')['price'].describe()
# no apparent relation
#df.groupby('minimum_nights')['price'].describe()
# no apparent relation

[ ]
 from matplotlib import pyplot as plt
_df_37['mean'].plot(kind='line', figsize=(8, 4), title='mean')
plt.gca().spines[['top', 'right']].set_visible(False)

In general, the mean price of the airbnb(s) seems to increase as their avaiability increases (overall trend, though with variations). This seems to suggest that airbnb(s) with higher availability, could be more likely to yield higher rents.

Double-click (or enter) to edit

Visualising numerical relationships
[ ]



[ ]
correlation_matrix = df[['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                       'calculated_host_listings_count','availability_365', 'price', 'latitude', 'longitude']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

The correlation matrix suggests that there is no strong correlation between any numerical features and rental prices.

[ ]

Note (after mid-project presentation): Change price to log price and see correlation matrix. And, add scatterplots instead of line graphs.

Visualising non-numerical relationships
[ ]
qualitative_columns = df.select_dtypes(exclude=['number']).columns
 # Handle missing values if needed

# Convert qualitative data to numerical representation using ordinal encoding
encoder = OrdinalEncoder()
df[qualitative_columns] = encoder.fit_transform(df[qualitative_columns])
print(df)
             id     name   host_id  host_name  neighbourhood_group  \
0          2539  12328.0      2787     4989.0                  1.0
1          2595  37455.0      2845     4785.0                  2.0
2          3647  43543.0      4632     2909.0                  2.0
3          3831  14783.0      4869     6203.0                  1.0
4          5022  18693.0      7192     5923.0                  2.0
...         ...      ...       ...        ...                  ...
48890  36484665  11647.0   8232441     9051.0                  1.0
48891  36485057   3520.0   6570630     6776.0                  1.0
48892  36485431  42464.0  23492952     4263.0                  2.0
48893  36485609   2572.0  30985759    10190.0                  2.0
48894  36487245  44538.0  68119814     1968.0                  2.0

       neighbourhood  latitude  longitude  room_type  price  minimum_nights  \
0              108.0  40.64749  -73.97237        1.0    149               1
1              127.0  40.75362  -73.98377        0.0    225               1
2               94.0  40.80902  -73.94190        1.0    150               3
3               41.0  40.68514  -73.95976        0.0     89               1
4               61.0  40.79851  -73.94399        0.0     80              10
...              ...       ...        ...        ...    ...             ...
48890           13.0  40.67853  -73.94995        1.0     70               2
48891           28.0  40.70184  -73.93317        1.0     40               4
48892           94.0  40.81475  -73.94867        0.0    115              10
48893           95.0  40.75751  -73.99112        2.0     55               1
48894           95.0  40.76404  -73.98933        1.0     90               7

       number_of_reviews  last_review  reviews_per_month  \
0                      9       1501.0               0.21
1                     45       1715.0               0.38
2                      0          NaN                NaN
3                    270       1760.0               4.64
4                      9       1532.0               0.10
...                  ...          ...                ...
48890                  0          NaN                NaN
48891                  0          NaN                NaN
48892                  0          NaN                NaN
48893                  0          NaN                NaN
48894                  0          NaN                NaN

       calculated_host_listings_count  availability_365
0                                   6               365
1                                   2               355
2                                   1               365
3                                   1               194
4                                   1                 0
...                               ...               ...
48890                               2                 9
48891                               2                36
48892                               1                27
48893                               6                 2
48894                               1                23

[48895 rows x 16 columns]
[ ]
correlation_matrix = df[['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                       'calculated_host_listings_count','availability_365', 'price', 'latitude', 'longitude','neighbourhood','room_type','neighbourhood_group']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

[ ]
# Separate numerical and categorical columns
numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(exclude=['number']).columns

# Handle missing values for numerical columns
imputer = SimpleImputer(strategy='mean')
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_columns]), columns=numeric_columns)
df_imputed = pd.concat([df_numeric_imputed, df[categorical_columns]], axis=1)
encoder = OrdinalEncoder()
encoded_data = encoder.fit_transform(df_imputed)
log_encoded_data = np.log(encoded_data + 1)  # Adding 1 to avoid log(0)
X = log_encoded_data[:, :-1]  # Features
y = log_encoded_data[:, -1]   # Target
threshold = np.mean(y)
binary_labels = (y > threshold).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, binary_labels, test_size=0.2, random_state=35)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Predict on the testing set
y_pred = model.predict(X_test)
feature_range=(min, max)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



[ ]
# Fit the model
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plot_roc_curve(model, X_test, y_test)
plt.title('ROC Curve')

# Plot Precision-Recall curve
plt.subplot(2, 2, 2)
plot_precision_recall_curve(model, X_test, y_test)
plt.title('Precision-Recall Curve')

# Plot Confusion Matrix
plt.subplot(2, 2, 3)
plot_confusion_matrix(model, X_test, y_test)
plt.title('Confusion Matrix')

# Plot Learning Curve
plt.subplot(2, 2, 4)
plot_learning_curve(model, X_train, y_train)
plt.title('Learning Curve')

plt.tight_layout()
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Predicted Negative', 'Predicted Positive'])
plt.yticks([0, 1], ['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted label')
plt.ylabel('True label')


train_sizes, train_scores, test_scores = learning_curve(
model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])

# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize model
model = RandomForestClassifier()

# Plot model performance
plot_model_performance(model, X_train, y_train, X_test, y_test)


[13]
11s
plt.figure(figsize=(10, 6))
mean_reviews_per_month = df.groupby('availability_365')['reviews_per_month'].mean().reset_index()
sns.boxplot(data=mean_reviews_per_month, x='availability_365', y='reviews_per_month') # corrected Avalibility to Availability
plt.title('Relationship between Availability and Reviews per month')
plt.xlabel('Availability')
plt.ylabel('Mean reviews per month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


