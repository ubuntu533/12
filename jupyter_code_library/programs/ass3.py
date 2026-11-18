import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv(r"C:\Users\Admin\Downloads\Admission_Predict (1).csv")
df.columns=df.columns.str.strip()

print("Original Dataset:")
print(df.head())

X=df.drop(["Serial No.","Chance of Admit"],axis=1)
y=(df["Chance of Admit"]>=0.5).astype(int)

features=X.columns

scaler=MinMaxScaler()
X=scaler.fit_transform(X)

Xtrain,Xtest,ytrain,ytest=train_test_split(
    X,y,test_size=0.25,random_state=42
)

model=DecisionTreeClassifier(max_depth=4,random_state=42)
model.fit(Xtrain,ytrain)

pred=model.predict(Xtest)

print("\nAccuracy:",accuracy_score(ytest,pred))
print("\nConfusion Matrix:")
print(confusion_matrix(ytest,pred))
print("\nClassification Report:")
print(classification_report(ytest,pred))

print("\nFeature Importance:")
for name,value in zip(features,model.feature_importances_):
    print(name,round(value,3))

plt.figure(figsize=(18,8))
plot_tree(model,feature_names=features,
          class_names=["Not Admitted","Admitted"],
          filled=True)
plt.show()