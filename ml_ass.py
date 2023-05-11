#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read in all the data
df = pd.read_csv(r"D:\FEB_2023\ML\Assignment\wine-quality-white-and-red.csv")


df['quality'].unique()

df.describe()
print(df['quality'].mode())

df.info()

df.shape

# show all available columns
df.columns

# check if got redundancy in columns
print(df.columns.tolist())

# Conclusion: There are no columns redundancy
# check for missing value
df.isna().sum()

# Conclusion: There are no missing values

# Renaming columns for better understanding
df.rename(columns={'quality': 'quality score'}, inplace=True)

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

table = pd.DataFrame({
    "No. Unique": df.nunique(),
    'NaN Value': df.isna().sum(),
    'Duplicated': df.duplicated().sum(),
    'Dtype': df.dtypes
})

# Check for duplicates
df.duplicated(keep='first')

duplicate_df = df[df.duplicated(keep='first')]

# Check for duplicated rows, keep the last occurrence as True, and mark the first occurrence as False

df = df.drop_duplicates(keep='first')

df.isna().sum()


table = pd.DataFrame({
    "No. Unique": df.nunique(),
    'NaN Value': df.isna().sum(),
    'Duplicated': df.duplicated().sum(),
    'Dtype': df.dtypes
})


ax = plt.figure(figsize=(10, 5))
ax = sns.countplot(x='quality score', data=df)
ax.bar_label(ax.containers[0])

ax = plt.figure(figsize=(15, 5))
ax = sns.countplot(x='quality score', hue='type', data=df)
for container in ax.containers:
    ax.bar_label(container)

df['type'].value_counts()
sns.barplot(x=df['quality score'], y=df['fixed acidity'])
sns.boxplot(x='fixed acidity', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['volatile acidity'])
sns.boxplot(x='volatile acidity', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['citric acid'])
sns.boxplot(x='citric acid', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['residual sugar'])
sns.boxplot(x='residual sugar', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['chlorides'])
sns.boxplot(x='chlorides', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['free sulfur dioxide'])
sns.boxplot(x='free sulfur dioxide', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['total sulfur dioxide'])
sns.boxplot(x='total sulfur dioxide', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['density'])
sns.boxplot(x='density', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['pH'])
sns.boxplot(x='pH', hue='quality score', data=df)

sns.barplot(x=df['quality score'], y=df['sulphates'])
sns.boxplot(x='sulphates', hue='quality score', data=df)
sns.barplot(x=df['quality score'], y=df['alcohol'])
sns.boxplot(x='alcohol', hue='quality score', data=df)

# Correlation between different variables
plt.figure(figsize=(16, 6))
sns.heatmap(df.corr(), annot=True)
plt.show()

df['type'].unique()


from sklearn.preprocessing import LabelEncoder

cols = ['type']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

df['type'].unique()

Y = df['quality score']
X = df.drop(columns='quality score')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=12345)

# calculate the median & convert it to integer
count = y_train.value_counts()
# n_samples = count.median().astype(np.int64)
n_samples = int(count.median())

# suppress warnings
import warnings

warnings.filterwarnings('ignore')

# define a utility function
def sampling_strategy(X, y, n_samples, t='majority'):
    target_classes = ''
    if t == 'majority':
        target_classes = y.value_counts() > n_samples
    elif t == 'minority':
        target_classes = y.value_counts() < n_samples
    tc = target_classes[target_classes == True].index
    # target_classes_all = y.value_counts().index
    sampling_strategy = {}
    for target in tc:
        sampling_strategy[target] = n_samples
    return sampling_strategy

# exploit the imblearn library
from imblearn.under_sampling import ClusterCentroids

under_sampler = ClusterCentroids(sampling_strategy=sampling_strategy(X_train, y_train, n_samples, t='majority'))
X_under, y_under = under_sampler.fit_resample(X_train, y_train)

from imblearn.over_sampling import SMOTE

over_sampler = SMOTE(sampling_strategy=sampling_strategy(X_under, y_under, n_samples, t='minority'), k_neighbors=2)
X_bal, y_bal = over_sampler.fit_resample(X_under, y_under)

# In[85]:


from collections import Counter

# summarize distribution
counter = Counter(y_bal)
for k, v in counter.items():
    per = v / len(y_bal) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier

# fit the split data into the model to make them learn based on the model
random_forest_model = RandomForestClassifier(random_state=12345)
random_forest_model.fit(X_train, y_train)



random_forest_predictions = random_forest_model.predict(X_test)
random_forest_model.score(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, random_forest_predictions))

print(confusion_matrix(y_test, random_forest_predictions))

# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# plot_confusion_matrix(random_forest_model, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(random_forest_model, X_test, y_test)
plt.title('Confusion Matrix')


from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y_train, random_forest_model.predict(X_train))
test_acc = accuracy_score(y_test, random_forest_predictions)

print(train_acc)
print(test_acc)

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': range(1, 10)}


random_forest_model_gs = RandomForestClassifier(random_state=12345)
random_forest_model_gs_best = GridSearchCV(estimator=random_forest_model_gs, param_grid=param_grid,
                                           return_train_score=True)
random_forest_model_gs_best.fit(X_train, y_train)

random_forest_model_gs_best_params = random_forest_model_gs_best.best_params_
print(random_forest_model_gs_best_params)

random_forest_predictions = random_forest_model_gs_best.predict(X_test)

print(classification_report(y_test, random_forest_predictions))

print(confusion_matrix(y_test, random_forest_predictions))

# plot_confusion_matrix(random_forest_model_gs_best, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(random_forest_model_gs_best, X_test, y_test)
plt.title('Confusion Matrix')

train_acc = accuracy_score(y_train, random_forest_model_gs_best.predict(X_train))
test_acc = accuracy_score(y_test, random_forest_predictions)

print(train_acc)
print(test_acc)

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(random_state=12345)
decision_tree_model.fit(X_train, y_train)

decision_tree_predictions = decision_tree_model.predict(X_test)

from sklearn import tree

tree.plot_tree(decision_tree_model)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, decision_tree_predictions))

print(confusion_matrix(y_test, decision_tree_predictions))

# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# plot_confusion_matrix(decision_tree_model, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(decision_tree_model, X_test, y_test)
plt.title('Confusion Matrix')

train_acc = accuracy_score(y_train, decision_tree_model.predict(X_train))
test_acc = accuracy_score(y_test, decision_tree_predictions)


from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': range(1, 10)}

decision_tree_model_gs = DecisionTreeClassifier(random_state=12345)
decision_tree_model_gs_best = GridSearchCV(estimator=decision_tree_model_gs, param_grid=param_grid,
                                           return_train_score=True)
decision_tree_model_gs_best.fit(X_train, y_train)

decision_tree_model_gs_best_params = decision_tree_model_gs_best.best_params_
print(decision_tree_model_gs_best_params)

decision_tree_predictions = decision_tree_model_gs_best.predict(X_test)

print(classification_report(y_test, decision_tree_predictions))

print(confusion_matrix(y_test, decision_tree_predictions))

# plot_confusion_matrix(decision_tree_model_gs_best, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(decision_tree_model_gs_best, X_test, y_test)
plt.title('Confusion Matrix')

train_acc = accuracy_score(y_train, decision_tree_model_gs_best.predict(X_train))
test_acc = accuracy_score(y_test, decision_tree_predictions)

print(train_acc)
print(test_acc)

# ## KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()  # default n_neighbors = 5
knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, knn_predictions))

print(confusion_matrix(y_test, knn_predictions))

# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# plot_confusion_matrix(knn_model, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test)
plt.title('Confusion Matrix')

train_acc = accuracy_score(y_train, knn_model.predict(X_train))
test_acc = accuracy_score(y_test, knn_predictions)

print(train_acc)
print(test_acc)

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(2, 10)}

knn_model_gs = KNeighborsClassifier()
knn_model_gs_best = GridSearchCV(estimator=knn_model_gs, param_grid=param_grid, return_train_score=True)
knn_model_gs_best.fit(X_train, y_train)

knn_model_gs_best_params = knn_model_gs_best.best_params_
print(knn_model_gs_best_params)

knn_predictions = knn_model_gs_best.predict(X_test)

print(classification_report(y_test, knn_predictions))

print(confusion_matrix(y_test, knn_predictions))

# plot_confusion_matrix(knn_model_gs_best, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(knn_model_gs_best, X_test, y_test)
plt.title('Confusion Matrix')

train_acc = accuracy_score(y_train, knn_model_gs_best.predict(X_train))
test_acc = accuracy_score(y_test, knn_predictions)

print(train_acc)
print(test_acc)

# ## Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(random_state=12345)  # default solver = 'lbfgs'
log_reg_model.fit(X_train, y_train)

log_reg_predictions = log_reg_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, log_reg_predictions))

print(confusion_matrix(y_test, log_reg_predictions))

# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# plot_confusion_matrix(log_reg_model, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(log_reg_model, X_test, y_test)
plt.title('Confusion Matrix')

train_acc = accuracy_score(y_train, log_reg_model.predict(X_train))
test_acc = accuracy_score(y_test, log_reg_predictions)

print(train_acc)
print(test_acc)

from sklearn.model_selection import GridSearchCV

param_grid = {'solver': ['newton-cg', 'liblinear', 'sag']}

log_reg_model_gs = LogisticRegression(random_state=12345)
log_reg_model_gs_best = GridSearchCV(estimator=log_reg_model_gs, param_grid=param_grid, return_train_score=True)
log_reg_model_gs_best.fit(X_train, y_train)
log_reg_model_gs_best_params = log_reg_model_gs_best.best_params_
print(log_reg_model_gs_best_params)

log_reg_predictions = log_reg_model_gs_best.predict(X_test)
print(classification_report(y_test, log_reg_predictions))
print(confusion_matrix(y_test, log_reg_predictions))
ConfusionMatrixDisplay.from_estimator(log_reg_model_gs_best, X_test, y_test)
plt.title('Confusion Matrix')

train_acc = accuracy_score(y_train, log_reg_model_gs_best.predict(X_train))
test_acc = accuracy_score(y_test, log_reg_predictions)

print(train_acc)
print(test_acc)

from sklearn.dummy import DummyClassifier

dclf = DummyClassifier(random_state=12345)
dclf.fit(X_train, y_train)

y_pred = dclf.predict(X_test)

train_acc = accuracy_score(y_train, dclf.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(train_acc)
print(test_acc)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# plot_confusion_matrix(dclf, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(dclf, X_test, y_test)
plt.title('Confusion Matrix')

# # Prediction

# expected to have 9 quality score

new = [[1, 7, 0.3, 0.35, 4, 0.03, 30, 110, 1, 3.3, 0.45, 11]]
new = sc.transform(new)
pred = knn_model.predict(new)
print(new, pred)

# expected to have 3 quality score

new = [[1, 7.9, 0.51, 0.27, 5, 0.079, 39, 120, 1, 3.2, 0.5, 10.1]]
new = sc.transform(new)
pred = knn_model.predict(new)
print(new, pred)
