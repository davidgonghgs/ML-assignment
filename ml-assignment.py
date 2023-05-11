import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read in all the data
# df = pd.read_csv(r"winequality.csv", sep = ",") # Reading semi-colon-delimited files in Pandas

# df.head(20)

# read in all the data
# Read the CSV file with semicolon delimiter and no header
# df = pd.read_csv(r"winequality-red.csv", delimiter=';') # Reading semi-colon-delimited files in Pandas


# Add column names to the DataFrame
# df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# df.head(20)

# read in all the data
# Read the first CSV file with semicolon delimiter
df = pd.read_csv(r"winequality-white.csv", delimiter=';') # Reading semi-colon-delimited files in Pandas

# Read the second CSV file with semicolon delimiter
df2 = pd.read_csv(r"winequality-red.csv", delimiter=';')

# Append the second DataFrame to the first one
df = pd.concat([df, df2], ignore_index=True)

df.head(20)

df.head()

df.sample(5)

df.tail()

df.dtypes

df['quality'].unique() # score between 0 and 10

df.describe()

print(df['quality'].mode())

df.info()

df.shape

df.columns

# check if got redundancy in columns
print(df.columns.tolist())

# check for missing value
df.isna().sum()

df.agg({
        'fixed acidity': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'volatile acidity': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'citric acid': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'residual sugar': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'chlorides': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'free sulfur dioxide': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'total sulfur dioxide': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'density': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'pH': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'sulphates': ['mean', 'median', 'min', 'max', 'std', 'skew'],
        'alcohol': ['mean', 'median', 'min', 'max', 'std', 'skew']
    })

fig = plt.figure(figsize = (15, 20))
ax = fig.gca()
df.hist(ax = ax)


# # Check for duplicates
# #df.duplicated(keep = False)
# df.duplicated(keep = False, subset=df.columns.difference(['quality']))

# Check for duplicated rows, keep the last occurrence as True, and mark the first occurrence as False
duplicates = df.duplicated(keep='first')

# Update the original DataFrame with a new column indicating if a row is duplicated or not
df['Duplicated'] = duplicates

df = df.drop_duplicates()

# Print the DataFrame with the 'Duplicated' column
print(df)

#duplicate = df[df.duplicated(keep = 'last')]
df = df[df.duplicated(subset=df.columns.difference(['quality']))]
duplicate = df
duplicate.shape

# access the data values fitted in the particular row or column based on the index value passed to the function
print(df.loc[[234]])

print(df.loc[[4783]])

print(df.loc[[2]])

# Check for missing values
df.isna().sum()


# Summary
table = pd.DataFrame({
    "No. Unique" : df.nunique(),
    'NaN Value': df.isna().sum(),
    'Duplicated' : df.duplicated().sum(),
    'Dtype': df.dtypes
})

table

# Check for imbalance target
ax = plt.figure(figsize = (10, 5))
ax = sns.countplot(x = 'quality', data = df)
ax.bar_label(ax.containers[0])