ASSIGNMENT 1 (ML)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("heart.csv")

# A. Cleaning
df.drop_duplicates(inplace=True)

# B. Missing Data
df.fillna(df.mean(numeric_only=True),inplace=True)
df.fillna(df.mode().iloc[0],inplace=True)

# C. Transformation
df=pd.get_dummies(df,drop_first=True,dtype=int)

# D. 75% Train, 25% Test
train,test=train_test_split(df,test_size=.25,random_state=42)

sc=MinMaxScaler()
train[train.columns]=sc.fit_transform(train)
test[test.columns]=sc.transform(test)

print("Training data:")
print(train.head())

print("\nTesting data:")
print(test.head())

print("\nTraining size:",train.shape)
print("Testing size:",test.shape)