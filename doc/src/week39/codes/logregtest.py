import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
credit = pd.read_csv('creditcard.csv')
print(credit.columns)
print(credit.head())
print("dimension of credit data: {}".format(credit.shape))
print(credit.groupby('Outcome').size())
#sns.countplot(credit[‘Outcome’],label=”Count”)
print(credit.info())
