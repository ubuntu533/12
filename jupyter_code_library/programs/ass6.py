

import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

# a) Preprocessing
data=pd.read_csv(
 r"C:\Users\Admin\Downloads\SMSSpamCollection",
 sep="\t",names=["label","message"])

data["label"]=data["label"].map({"ham":0,"spam":1})

cv=CountVectorizer()
x=cv.fit_transform(data["message"])
y=data["label"]

# b) Train-Test Split
xtrain,xtest,ytrain,ytest=train_test_split(
 x,y,test_size=0.2,random_state=42)

# c) Two ML Algorithms
m1=MultinomialNB().fit(xtrain,ytrain)
m2=BernoulliNB().fit(xtrain,ytrain)

print("MNB Accuracy:",m1.score(xtest,ytest))
print("BNB Accuracy:",m2.score(xtest,ytest))

# d) Cross Validation
print("MNB CV:",cross_val_score(m1,x,y,cv=5).mean())
print("BNB CV:",cross_val_score(m2,x,y,cv=5).mean())

# e) Hyperparameter Tuning
p={"alpha":[0.1,0.5,1]}

g1=GridSearchCV(m1,p,cv=5).fit(xtrain,ytrain)
g2=GridSearchCV(m2,p,cv=5).fit(xtrain,ytrain)

print("Tuned MNB:",g1.score(xtest,ytest))
print("Tuned BNB:",g2.score(xtest,ytest))
print("Best MNB:",g1.best_params_)
print("Best BNB:",g2.best_params_)