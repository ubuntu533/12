import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def evaluate_model(TP,FP,FN,TN):

    y_true=np.array([1]*TP+[0]*FP+[1]*FN+[0]*TN)
    y_pred=np.array([1]*TP+[1]*FP+[0]*FN+[0]*TN)

cm=confusion_matrix(y_true,y_pred)
accuracy=accuracy_score(y_true,y_pred)
precision=precision_score(y_true,y_pred)
recall=recall_score(y_true,y_pred)
f1=f1_score(y_true,y_pred)

print("Confusion Matrix:\n",cm)
print(f"Accuracy:{accuracy:.2f}")
print(f"Precision:{precision:.2f}")
print(f"Recall:{recall:.2f}")
print(f"F1 Score:{f1:.2f}")

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["Pred Neg","Pred Pos"],
            yticklabels=["Actual Neg","Actual Pos"])
plt.title("Confusion Matrix")
plt.show()

evaluate_model(45, 55, 5, 395)
