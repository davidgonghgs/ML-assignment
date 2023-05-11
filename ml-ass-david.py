import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

import seaborn as sns

import pickle

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('winequality.csv')



# look at the first five rows of the dataset.
print("------------------------------------")
print("First five rows of the dataset: ")
print(df.head())

# check data set info - explore the type of data present in each of the columns present in the dataset.
print("------------------------------------")
print("Data set info: ")
df.info()

# explore the descriptive statistical measures of the dataset.
print("------------------------------------")
print("Descriptive statistical measures of the dataset: ")
print(df.describe().T)

# check the number of null values in the dataset columns wise.
print("------------------------------------")
print("Number of null values: ", df.isnull().sum().sum())

# impute the missing values by means as the data present in the different columns are continuous values.
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

print("Number of null values after imputation: ", df.isnull().sum().sum())

# draw the histogram to visualise the distribution of the data with continuous values in the columns of the dataset.
df.hist(bins=20, figsize=(10, 10))
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# 6 Relationship between features & the target
# check how the quality is influenced by fixed acidity
ax = sns.barplot(x=df['quality'], y=df['fixed acidity'])
ax.set_title("the quality is influenced by fixed acidity")
plt.show()

# check how the quality is influenced by volatile acidity
ax = sns.barplot(x=df['quality'], y=df['volatile acidity'])
ax.set_title("the quality is influenced by volatile acidity")
plt.show()

# check how the quality is influenced by citric acid
ax = sns.barplot(x=df['quality'], y=df['citric acid'])
ax.set_title("the quality is influenced by citric acid")
plt.show()

# check how the quality is influenced by residual sugar
ax = sns.barplot(x=df['quality'], y=df['residual sugar'])
ax.set_title("the quality is influenced by residual sugar")
plt.show()

# check how the quality is influenced by chlorides
ax = sns.barplot(x=df['quality'], y=df['chlorides'])
ax.set_title("the quality is influenced by chlorides")
plt.show()

# check how the quality is influenced by free sulfur dioxide
ax = sns.barplot(x=df['quality'], y=df['free sulfur dioxide'])
ax.set_title("the quality is influenced by free sulfur dioxide")
plt.show()

# check how the quality is influenced by total sulfur dioxide
ax = sns.barplot(x=df['quality'], y=df['total sulfur dioxide'])
ax.set_title("the quality is influenced by total sulfur dioxide")
plt.show()

# check how the quality is influenced by density
ax = sns.barplot(x=df['quality'], y=df['density'])
ax.set_title("the quality is influenced by density")
plt.show()

# check how the quality is influenced by pH
ax = sns.barplot(x=df['quality'], y=df['pH'])
ax.set_title("the quality is influenced by pH")
plt.show()

# check how the quality is influenced by sulphates
ax = sns.barplot(x=df['quality'], y=df['sulphates'])
ax.set_title("the quality is influenced by sulphates")
plt.show()

# check how the quality is influenced by alcohol
ax = sns.barplot(x=df['quality'], y=df['alcohol'])
ax.set_title("the quality is influenced by alcohol")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

#  Relationship between features & the target - Correlation matrix
print("------------------------------------------")
print("Correlation matrix : ")
print(df.corr())

plt.figure(figsize=(16, 6))
sns.heatmap(df.corr(), annot=True);

# we need find some redundant features in our data set, which do not help us increasing the model's performance.
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()

# we replace the type, with the 0 and 1, as there are only two categories
df.replace({'white': 1, 'red': 0}, inplace=True)

# From the above heat map we can conclude that the ‘total sulphur dioxide’ and ‘free sulphur dioxide‘ are highly correlated features so, we will remove them.
lrdf = df.copy()
# df = df.drop(['total sulfur dioxide', 'free sulfur dioxide'], axis=1)
lrdf = lrdf.drop(['total sulfur dioxide', 'free sulfur dioxide'], axis=1)

# ------------------------------------------------------------------------------------------------------------------------
# prepare our data for training and splitting it into training and validation data so, that we can select which model’s performance is best as per the use case.
# df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
lrdf['best quality'] = [1 if x > 5 else 0 for x in lrdf.quality]



# get a copy of the data set for logical regression

# After segregating features and the target variable from the dataset we will split it into 80:20 ratio for model selection.
features = lrdf.drop(['quality', 'best quality'], axis=1)
target = lrdf['best quality']


# ------------------------------------------------------------------------------------------------------------------------
print("------------------LogisticRegression:------------------------")
# Splitting the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

print("------------------------------------------")
print("Shape of the training and testing data : ")
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

# Normalising the data before training help us to achieve stable and fast training of the model. LogisticRegression()
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

print("Model Evaluation : ")
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
for i in range(3):
    models[i].fit(xtrain, ytrain)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print()

# From the above accuracies we can say that Logistic Regression and SVC() classifier performing better
# on the validation data with less difference between the validation and training data.
# Let’s plot the confusion matrix as well for the validation data using the Logistic Regression model.
metrics.plot_confusion_matrix(models[1], xtest, ytest)
plt.show()

#print the classification report for the best performing model.
print("------------------------------------------")
print("Classification Report : ")
print(metrics.classification_report(ytest, models[1].predict(xtest)))



# use linear regression to predict the quality of the wine.
print("-------------------Linear Regression:-----------------------")

# get a copy of the data set for linear regression
lineardf = df.copy()
X = lineardf.drop('quality', axis=1)
y = lineardf['quality']

# split the features and target data sets into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('lm_model1 train/test shapes:')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# create and fit a linear regression model
lm_wine_1 = LinearRegression()
lm_model1 = lm_wine_1.fit(X_train, y_train)

# computing yhat (ie train_predictions) using X (ie train_features)
train_predictions = lm_wine_1.predict(X_train)
train_prediction = [int(round(x, 0)) for x in train_predictions]

# simple function to compare actual and predicted values
def compare_prediction(y, yhat):
    comp_matrix = pd.DataFrame(zip(y_train,train_prediction), columns = ['Actual', 'Predicted'])
    comp_matrix['Err'] = abs(comp_matrix['Actual']-comp_matrix['Predicted'])
    comp_matrix['PctErr'] = comp_matrix['Err']/comp_matrix['Actual'] * 100
    mean_value = np.mean(comp_matrix['PctErr'])
    return comp_matrix, mean_value

# compare actual and predicted values
comp_matrix, mean = compare_prediction(y, train_prediction)
print("lm_model1 prediction comparison and mean error:", comp_matrix, mean)

accuracy1 = round((100-mean),2)
print('lm_model1 accuracy =', accuracy1)
# 90.66% accuracy


print("-------------------Linear Regression 2 :-----------------------")
# if follow the heat map to remove 'total sulfur dioxide', 'free sulfur dioxide'
lineardf = df.copy()
lst = ['total sulfur dioxide', 'free sulfur dioxide']
lineardf.drop(lst, axis =1, inplace = True)
X = lineardf.drop('quality', axis=1)
y = lineardf['quality']

# split the features and target data sets into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('lm_model2 train/test shapes:')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
lm_wine2 = LinearRegression()
lm_model2 = lm_wine2.fit(X_train, y_train)

# computing yhat (ie train_predictions) using X (ie train_features)
train_predictions = lm_wine2.predict(X_train)
train_prediction = [int(round(x,0)) for x in train_predictions]

comp_matrix, mean = compare_prediction(y, train_prediction)
print("lm_model2 prediction comparison and mean error:", comp_matrix, mean)

accuracy2 = round((100-mean), 2)
print("lm_model2 accuracy =", accuracy2)

# 90.58% accuracy



print("-------------------Linear Regression 3 :-----------------------")
# Drop columns showing weak correlations (0.2 - 0.4)
# Drop columns fixed acidity, citric acid, chlorides, total sulfur dioxide, density.
lineardf = df.copy()
lst = ['fixed acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'density']
lineardf.drop(lst, axis =1, inplace = True)
X = lineardf.drop('quality', axis=1)
y = lineardf['quality']

# split the features and target data sets into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('lm_model3 train/test shapes:')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# create and fit a linear regression model
lm_wine3 = LinearRegression()
lm_model3 = lm_wine3.fit(X_train, y_train)

# computing yhat (ie train_predictions) using X (ie train_features)
train_predictions = lm_wine3.predict(X_train)
train_prediction = [int(round(x,0)) for x in train_predictions]

# compare actual and predicted values
comp_matrix, mean = compare_prediction(y, train_prediction)
print("lm_model3 prediction comparison and mean error:", comp_matrix, mean)

accuracy3 = round((100-mean), 2)
print("lm_model3 accuracy =", accuracy3)
# 90.65% accuracy

print("Model1 Accuracy {}".format(accuracy1))
print("Model2 Accuracy {}".format(accuracy2))
print("Model3 Accuracy {}".format(accuracy3))

model_file = open('wine_model.pkl', 'wb')
pickle.dump(obj=lm_model1, file=model_file)
model_file.close()

model_file = open('wine_model.pkl', 'rb')
lr_model = pickle.load(model_file)
model_file.close()
print(lr_model)