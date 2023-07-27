import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

df = pd.read_csv('churn_clean.csv')

#perform a t-test using scipy.stats.ttest_ind on the 'Tenure' and 'Churn' columns
x = ttest_ind(df[df['Churn'] == 'Yes']['Outage_sec_perweek'], df[df['Churn'] == 'No']['Outage_sec_perweek'])

#Return our our T-test results.
#The first value is the difference between means of the two groups (mean(a) - mean(b)) divided by the standard error.
#The second value is the p-value
print(x)
if x[1] > .05:
    print('P value is above .05, so we are unable reject the null hypothesis')
else:
    print('P value is below .05, so we can reject the null hypothesis')

#A p value of .9875 is greater than .05, so we fail to reject the null hypothesis.

#Look at the distribution of 'Outage_sec_perweek' column for churned and non-churned customers
df[df['Churn'] == 'Yes']['Outage_sec_perweek'].plot(kind='hist', rot=0, bins=30, title='Outage_sec_perweek for Churned Customers')
plt.show()
df[df['Churn'] == 'No']['Outage_sec_perweek'].plot(kind='hist', rot=0, bins=30, title='Outage_sec_perweek for Non-Churned Customers')
plt.show()

#Get the standard deviation of the 'Outage_sec_perweek' column for churned and non-churned customers
print("Churned Standard deviation: " + str(df[df['Churn'] == 'Yes']['Outage_sec_perweek'].std()))
print("Non-Churned Standard deviation: " + str(df[df['Churn'] == 'No']['Outage_sec_perweek'].std()))

#Box plots of the 'Age' and 'Children' columns
df['Age'].plot(kind='box', rot=0)
plt.show()
df['Children'].plot(kind='box', rot=0)
plt.show()

#Histogram of the 'Age' column
df['Age'].plot(kind='hist', rot=0, bins=20, title='Age')
plt.show()

#Bar charts of the InternetService and Marital columns
df['InternetService'].value_counts().plot(kind='bar', rot=0, title='Internet Service').invert_xaxis()
plt.show()
df['Marital'].value_counts().plot(kind='bar', rot=0, title='Marital Status')
plt.show()

#make a correlation plot out of the 'Age' and 'Children' columns
df[['Age', 'Children']].plot(kind='scatter', x='Age', y='Children', alpha=.01)
plt.show()

#Make a stacked bar chart out of the Gender and Techie columns
crosstab_01 = pd.crosstab(df['Gender'], df['Techie']).sort_index(axis=1, ascending=False)
crosstab_01.plot(kind='bar', stacked = True, rot=0)
crosstab_norm = crosstab_01.div(crosstab_01.sum(1), axis = 0)
plt.show()
crosstab_norm.plot(kind='bar', stacked = True, rot=0, title="Normalized Ratios of Techie vs. Non-Techie by Gender")
plt.show()

exit()