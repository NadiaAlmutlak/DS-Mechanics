import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df=pd.read_csv('/Users/nadiam/2-DSforMechanics/Homework 1- Boston Housing/Boston_Housing_Data.csv')
#print(df.describe().to_string())


#1. There is no missing values as shown by the describe command
#2. summary of the statistics are shown in the output of describe

# Generate a correlation graph
corr=df.corr()
ax=sns.heatmap(corr,annot=True, linewidths=.5)
plt.show()

#22.532806 is the median home value
#Some positive correlations are room size (0.7), zones (0.3), dist (0.25)
#Some negative correlations are nox (-0.43), age (-0.38)

#Explore room number against med value of home
room_reg=sns.regplot(x='rm', y='medv', data=df)
plt.show()
df.groupby('rm').agg('mean')[['medv']].plot(kind='bar',stacked=False,color=("lightcoral"))
plt.show()
#We can see that room size correaltes with med value

#Explore zone against med value of home
zone_reg=sns.regplot(x='zn', y='medv', data=df)
plt.show()

#The data seems skewed here

#nox relation
nox_reg=sns.regplot(x='nox', y='medv', data=df)
plt.show()

#turning room number into a numerical feature
