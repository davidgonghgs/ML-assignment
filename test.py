import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create a DataFrame
red_wine_df = pd.read_csv('winequality-red.csv', sep=';')
white_wine_df = pd.read_csv('winequality-white.csv', sep=';')

plt.plot(bins1, hist1_new, label='red_quality')
plt.plot(bins2, hist2_new, label='white_quality')

# drow the frequency graph of the quality for both red and white wine to compare
sns.countplot(red_wine_df['quality'])
plt.show()

sns.countplot(white_wine_df['quality'])
plt.show()
