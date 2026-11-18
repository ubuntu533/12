import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load Dataset
df = pd.read_csv(r"C:\Users\Admin\Downloads\Mall_Customers.csv")
print("Original Dataset")
print(df.head())

# Data Preprocessing
df = df.drop_duplicates()
df = df.dropna()

le = LabelEncoder()
df["Genre"] = le.fit_transform(df["Genre"])

X = df.iloc[:, [3, 4]]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_train)
kmeans_labels = kmeans.labels_

print("K-Means Silhouette Score:", silhouette_score(X_train, kmeans_labels))

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=5)
hc.fit(X_train)
hc_labels = hc.labels_

print("Hierarchical Silhouette Score:", silhouette_score(X_train, hc_labels))

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans_labels)
plt.title("KMeans Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.subplot(1, 2, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=hc_labels)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()