import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df=pd.read_csv('/Users/nadiam/2-DSforMechanics/Homework 1- Boston Housing/Boston_Housing_Data.csv')

print(df['rm'].round())
# Assuming rooms is the most correlated feature for an accurate price prediction

#Let's assume we have a home with this number of rooms, we need to find the euclidean distance between it
#and a bunch of other rooms and explore the data

room_value = 8
first_room_value = df.loc[0,'rm']
first_distance = np.abs(first_room_value - room_value)
print(first_distance)

df['distance'] = np.abs(df.rm.round() - room_value)
print(df.distance.value_counts().sort_index())

#lets get some random samples
df_rooms = df.sample(frac=1,random_state=0)
df_rooms = df_rooms.sort_values('distance')
print("4 other homes of a similar size have a median value of", df_rooms.medv.head())

# we get the mean price of our last set
mean_price = df_rooms.medv.iloc[:5].mean()
print("Our predicted med value for a home of this size is", mean_price)